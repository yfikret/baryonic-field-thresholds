import numpy as np
from scipy.optimize import lsq_linear

# Units: kpc, km/s, Msun
# Gives acceleration in (km/s)^2 / kpc.

from src.constants import *

def annulus_edges(r):
    r = np.asarray(r, dtype=float)

    edges = np.empty(len(r) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

    return edges


def annulus_centroids(edges):
    rin = edges[:-1]
    rout = edges[1:]

    return (2.0 / 3.0) * (rout**3 - rin**3) / (rout**2 - rin**2)


def weighted_percentile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[good]
    w = w[good]

    if len(x) == 0:
        return np.nan

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    cw = np.cumsum(w)
    cw /= cw[-1]

    return np.interp(q, cw, x)


def choose_gas_softening(r, vgas=None):
    """
    Simple dynamic softening in kpc.

    For gas, use a gas-weighted high percentile of annulus widths.
    This avoids undersoftenening galaxies whose gas contribution lives
    mostly in sparsely sampled outer regions.
    """
    r = np.asarray(r, dtype=float)
    r = np.sort(r[np.isfinite(r) & (r > 0)])

    edges = np.empty(len(r) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

    widths = edges[1:] - edges[:-1]

    if vgas is not None:
        vgas = np.asarray(vgas, dtype=float)

        # Signed gas acceleration magnitude.
        # Weight the bins where gas dynamically matters.
        w = np.abs(vgas * np.abs(vgas) / r)

        width_eff = weighted_percentile(widths, w, 0.75)
    else:
        width_eff = np.percentile(widths, 75)

    if not np.isfinite(width_eff):
        width_eff = np.median(np.diff(r))

    eps_grav = max(0.05, 0.75 * width_eff)
    
    # Prevent pathological over-softening.
    eps_grav = min(eps_grav, 0.12 * r[-1])

    return {
        "eps_grav": eps_grav,
        "eps_kernel": max(0.15, 2.0 * eps_grav),
    }

def choose_disk_softening(r, sb_disk=None, vdisk=None):
    """
    Disk-specific gravitational softening in kpc.

    Simpler and usually smaller than the gas softening.
    Uses the annulus widths weighted by where the stellar disk matters.
    """
    r = np.asarray(r, dtype=float)

    order = np.argsort(r)
    r = r[order]

    good = np.isfinite(r) & (r > 0)
    r = r[good]

    edges = np.empty(len(r) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

    widths = edges[1:] - edges[:-1]
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    if vdisk is not None:
        vdisk = np.asarray(vdisk, dtype=float)[order][good]

        # Weight where the disk contribution to acceleration matters.
        w = np.maximum(vdisk, 0.0)**2 / r

    elif sb_disk is not None:
        sb_disk = np.asarray(sb_disk, dtype=float)[order][good]
        sb_disk = np.nan_to_num(sb_disk, nan=0.0, posinf=0.0, neginf=0.0)

        # Luminosity per annulus proxy.
        w = np.maximum(sb_disk, 0.0) * area

    else:
        w = np.ones_like(r)

    width_eff = weighted_percentile(widths, w, 0.55)

    if not np.isfinite(width_eff):
        width_eff = np.median(np.diff(r))

    disk_eps_coeff = 1.50
    disk_eps_floor = 0.09
    disk_eps_cap_frac = 0.06

    # Disk should usually be sharper than gas.
    eps_grav = max(disk_eps_floor, disk_eps_coeff * width_eff)

    # Avoid over-softening compact stellar disks.
    eps_grav = min(eps_grav, disk_eps_cap_frac * r[-1])

    return {
        "eps_grav": eps_grav,
        "eps_kernel": max(0.15, 2.0 * eps_grav),
    }

def choose_bulge_softening(r):
    r = np.asarray(r, dtype=float)
    r = np.sort(r[np.isfinite(r) & (r > 0)])

    dr_med = np.median(np.diff(r))

    eps_grav = max(0.005, 0.15 * dr_med)
    eps_grav = min(eps_grav, 0.015 * r[-1])

    return {
        "eps_grav": eps_grav,
        "eps_kernel": max(0.05, 3.0 * eps_grav),
    }

def smooth_121(y, passes=1):
    """
    Simple, explainable 1-2-1 smoothing.
    Useful for noisy gas velocity targets.
    """
    y = np.asarray(y, dtype=float).copy()

    for _ in range(passes):
        if len(y) < 3:
            return y

        z = y.copy()
        z[1:-1] = 0.25 * y[:-2] + 0.50 * y[1:-1] + 0.25 * y[2:]
        y = z

    return y


def second_difference_matrix(n):
    if n < 3:
        return np.zeros((0, n))

    D = np.zeros((n - 2, n))

    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    return D


def rings_from_annuli(
    ring_r,
    annulus_m,
    target_arc_kpc=0.08,
    nphi_min=96,
    nphi_max=1800,
):
    """
    Convert annular masses into equal-mass azimuthal elements.

    This is deliberately simple: each annulus becomes a ring of point masses.
    """
    xs, ys, zs, ms = [], [], [], []
    golden = (np.sqrt(5.0) - 1.0) / 2.0

    for i, (R, M) in enumerate(zip(ring_r, annulus_m)):
        if not np.isfinite(M) or M <= 0.0:
            continue

        nphi = int(np.ceil(2.0 * np.pi * max(R, target_arc_kpc) / target_arc_kpc))
        nphi = int(np.clip(nphi, nphi_min, nphi_max))

        # Ring phase offset prevents repeated alignment with the +x sampling axis.
        phase = (i * golden) % 1.0

        j = np.arange(nphi)
        phi = 2.0 * np.pi * (j + 0.5 + phase) / nphi

        xs.append(R * np.cos(phi))
        ys.append(R * np.sin(phi))
        zs.append(np.zeros(nphi))
        ms.append(np.full(nphi, M / nphi))

    if not xs:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([]),
            "m": np.array([]),
        }

    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "z": np.concatenate(zs),
        "m": np.concatenate(ms),
    }


def ring_response_matrix(r_eval, ring_r, eps_grav, target_arc_kpc=0.08):
    """
    Response matrix from annular ring masses to radial acceleration.

    Rmat[i, j] is g_rad at r_eval[i] from one Msun in ring j.
    """
    r_eval = np.asarray(r_eval, dtype=float)
    ring_r = np.asarray(ring_r, dtype=float)

    Rmat = np.zeros((len(r_eval), len(ring_r)))

    for j, R in enumerate(ring_r):
        ring = rings_from_annuli(
            [R],
            [1.0],
            target_arc_kpc=target_arc_kpc,
            nphi_min=160,
            nphi_max=2200,
        )

        x = ring["x"]
        y = ring["y"]
        m = ring["m"]

        for i, r0 in enumerate(r_eval):
            dx = x - r0
            dy = y

            d2 = dx**2 + dy**2 + eps_grav**2
            d3 = d2 * np.sqrt(d2)

            gx = G_KPC * np.sum(m * dx / d3)

            # Same sign convention as your compute_profiles_from_elements:
            # inward radial acceleration at +x is -gx.
            Rmat[i, j] = -gx

    return Rmat


def fit_annular_masses(
    r,
    g_target,
    eps_grav,
    sigma_prior=None,
    smooth_strength=0.8,
    prior_strength=0.0,
    target_arc_kpc=0.08,
):
    """
    Fit non-negative annular masses to a target radial acceleration curve.

    If sigma_prior is provided, the unknown is a smooth multiplicative
    correction to that prior.

    If sigma_prior is not provided, the unknown is the surface density itself.
    """
    r = np.asarray(r, dtype=float)
    g_target = np.asarray(g_target, dtype=float)

    edges = annulus_edges(r)
    ring_r = annulus_centroids(edges)
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    Rmat = ring_response_matrix(
        r,
        ring_r,
        eps_grav=eps_grav,
        target_arc_kpc=target_arc_kpc,
    )

    bscale = np.percentile(np.abs(g_target[np.isfinite(g_target)]), 90)
    bscale = max(bscale, 1e-12)

    n = len(r)
    D2 = second_difference_matrix(n)

    if sigma_prior is not None:
        sigma_prior = np.asarray(sigma_prior, dtype=float)
        sigma_prior = np.nan_to_num(sigma_prior, nan=0.0, posinf=0.0, neginf=0.0)

        floor = 1e-6 * np.nanmax(sigma_prior) if np.nanmax(sigma_prior) > 0 else 1.0
        sigma_prior = np.maximum(sigma_prior, floor)

        # Unknown u satisfies:
        #     Sigma = sigma_prior * u
        A_main = Rmat * area[None, :] * sigma_prior[None, :] / bscale
        b_main = g_target / bscale

        rows = [A_main]
        rhs = [b_main]

        if smooth_strength > 0:
            rows.append(smooth_strength * D2)
            rhs.append(np.zeros(D2.shape[0]))

        if prior_strength > 0:
            rows.append(prior_strength * np.eye(n))
            rhs.append(prior_strength * np.ones(n))

        A_fit = np.vstack(rows)
        b_fit = np.concatenate(rhs)

        result = lsq_linear(
            A_fit,
            b_fit,
            bounds=(0.0, np.inf),
            lsmr_tol="auto",
            verbose=0,
        )

        sigma = sigma_prior * result.x

    else:
        A_surface = Rmat * area[None, :]

        sigma_scale = bscale / max(np.max(np.abs(A_surface)), 1e-30)

        A_main = A_surface * sigma_scale / bscale
        b_main = g_target / bscale

        rows = [A_main]
        rhs = [b_main]

        if smooth_strength > 0:
            rows.append(smooth_strength * D2)
            rhs.append(np.zeros(D2.shape[0]))

        A_fit = np.vstack(rows)
        b_fit = np.concatenate(rhs)

        result = lsq_linear(
            A_fit,
            b_fit,
            bounds=(0.0, np.inf),
            lsmr_tol="auto",
            verbose=0,
        )

        sigma = sigma_scale * result.x

    masses = sigma * area

    return ring_r, masses


def fibonacci_shell(radius, mass, n=512):
    """
    Nearly uniform spherical shell.
    Used for the bulge.
    """
    if mass <= 0.0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([]),
            "m": np.array([]),
        }

    k = np.arange(n)

    z = 1.0 - 2.0 * (k + 0.5) / n
    theta = np.pi * (1.0 + np.sqrt(5.0)) * k
    xy = np.sqrt(1.0 - z**2)

    return {
        "x": radius * xy * np.cos(theta),
        "y": radius * xy * np.sin(theta),
        "z": radius * z,
        "m": np.full(n, mass / n),
    }



def bulge_shells_from_vbul(r, vbul, ml_bulge=1.0, n_shell=512):
    """
    Spherical bulge approximation from SPARC Vbul.

    M(<r) = r Vbul^2 / G.
    """
    r = np.asarray(r, dtype=float)
    vbul = np.asarray(vbul, dtype=float)

    pieces = []

    menc = r * ml_bulge * vbul**2 / G_KPC
    
    M_core = menc[0]
    pieces.append(fibonacci_shell(0.35 * r[0], M_core, n=n_shell))

    menc = np.maximum(menc - M_core, 0.0)

    menc = np.maximum.accumulate(menc)

    shell_m = np.diff(np.concatenate([[0.0], menc]))

    for R, M in zip(r, shell_m):
        if M > 0:
            pieces.append(fibonacci_shell(R, M, n=n_shell))

    
    return combine_elements(pieces)


def combine_elements(pieces):
    pieces = [p for p in pieces if p is not None and len(p["m"]) > 0]

    if not pieces:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([]),
            "m": np.array([]),
        }

    return {
        "x": np.concatenate([p["x"] for p in pieces]),
        "y": np.concatenate([p["y"] for p in pieces]),
        "z": np.concatenate([p["z"] for p in pieces]),
        "m": np.concatenate([p["m"] for p in pieces]),
    }


def build_sparc_elements_v2(
    df,
    ml_disk,
    ml_bulge,
    include_disk=True,
    include_gas=True,
    include_bulge=True,
    gas_smoothing_passes=2,
    disk_smooth_strength=0.5,
    gas_smooth_strength=2.0,
    disk_prior_strength=0.05,
    target_arc_kpc=0.08,
):
    """
    Build mass elements for compute_profiles_from_elements.

    This version is calibrated to the SPARC velocity columns.

    Returns:
        elements, r_samples, eps_info

    Use with:
        compute_profiles_from_elements(..., G=G_KPC)
    """
    r = np.asarray(df["Rad"], dtype=float)
    vgas = np.asarray(df["Vgas"], dtype=float)
    vdisk = np.asarray(df["Vdisk_unity"], dtype=float)
    vbul = np.asarray(df["Vbul_unity"], dtype=float)
    sb_disk = np.asarray(df["SBdisk"], dtype=float)
    
    order = np.argsort(r)

    r = r[order]
    vgas = vgas[order]
    vdisk = vdisk[order]
    vbul = vbul[order]
    sb_disk = sb_disk[order]

    good = np.isfinite(r) & (r > 0)

    r = r[good]
    vgas = np.nan_to_num(vgas[good], nan=0.0)
    vdisk = np.nan_to_num(vdisk[good], nan=0.0)
    vbul = np.nan_to_num(vbul[good], nan=0.0)
    sb_disk = np.nan_to_num(sb_disk[good], nan=0.0, posinf=0.0, neginf=0.0)

    #eps_info = choose_softening(r)

    eps_info_gas = choose_gas_softening(r, vgas)

    eps_info_disk = choose_disk_softening(
    r,
    sb_disk=sb_disk,
    vdisk=vdisk,
    )

    eps_info_bulge = choose_bulge_softening(r)

    eps_info = eps_info_bulge

    disk_pieces = []
    gas_pieces = []
    bulge_pieces = []

    if include_disk:
        # Disk target acceleration.
        # SPARC Vdisk is for M/L = 1, so multiply V^2 by ml_disk.
        g_disk = ml_disk * vdisk**2 / r
        
        # SBdisk is Lsun / pc^2.
        # Convert to Msun / kpc^2 using M/L and 1 kpc^2 = 1e6 pc^2.
        sigma_disk_prior = ml_disk * sb_disk * 1e6

        ring_r, disk_m = fit_annular_masses(
            r,
            g_disk,
            eps_grav=eps_info_disk["eps_grav"],
            sigma_prior=sigma_disk_prior,
            smooth_strength=disk_smooth_strength,
            prior_strength=disk_prior_strength,
            target_arc_kpc=target_arc_kpc,
        )

        disk_pieces.append(
            rings_from_annuli(
                ring_r,
                disk_m,
                target_arc_kpc=target_arc_kpc,
            )
        )

        disk_elements = combine_elements(disk_pieces)
    else:
        disk_elements = None

    if include_gas:
        # Signed gas contribution.
        # Negative central Vgas values are meaningful in SPARC.
        g_gas = vgas * np.abs(vgas) / r
        g_gas = smooth_121(g_gas, passes=gas_smoothing_passes)

        ring_r, gas_m = fit_annular_masses(
            r,
            g_gas,
            eps_grav=eps_info_gas["eps_grav"],
            sigma_prior=None,
            smooth_strength=gas_smooth_strength,
            prior_strength=0.0,
            target_arc_kpc=target_arc_kpc,
        )

        gas_pieces.append(
            rings_from_annuli(
                ring_r,
                gas_m,
                target_arc_kpc=target_arc_kpc,
            )
        )

        gas_elements = combine_elements(gas_pieces)
    else:
        gas_elements = None

    if include_bulge and np.any(vbul > 0):
        bulge_pieces.append(
            bulge_shells_from_vbul(
                r,
                vbul,
                ml_bulge=ml_bulge,
                n_shell=512,
            )
        )
        bulge_elements = combine_elements(bulge_pieces)
    else:
        bulge_elements = None
    
    elements = {"disk": disk_elements, "bulge": bulge_elements, "gas": gas_elements}
    eps_info = {"disk": eps_info_disk, "bulge": eps_info_bulge, "gas": eps_info_gas}

    return elements, r, eps_info