import numpy as np
import pickle
import os
from pathlib import Path

from src.element_builder import build_sparc_elements_v2

from src.constants import *

def compute_profiles_from_elements(
    elements,
    r_samples,
    eps_info,
    embedding_power=2.0,
    G=G_METER,
):

    x = elements["x"]
    y = elements["y"]
    z = elements["z"]
    m = elements["m"]

    if isinstance(eps_info, dict):
        eps_grav = float(eps_info["eps_grav"])
        eps_kernel = float(eps_info["eps_kernel"])
    else:
        # Backward compatibility.
        eps_grav = float(eps_info)
        eps_kernel = float(eps_info)

    g_bar = []
    T_vals = []
    A_vals = []
    Ax_vals = []
    Ay_vals = []
    Az_vals = []

    negative_g_count = 0

    for r0 in r_samples:
        dxp = x - r0
        dyp = y
        dzp = z

        d2_phys = dxp**2 + dyp**2 + dzp**2
        d_phys = np.sqrt(d2_phys)

        # Gravity softening.
        d2_grav = d2_phys + eps_grav**2
        d_grav = np.sqrt(d2_grav)
        d3_grav = d2_grav * d_grav

        gx = G * np.sum(m * dxp / d3_grav)
        gy = G * np.sum(m * dyp / d3_grav)
        gz = G * np.sum(m * dzp / d3_grav)

        # Radial direction is +x, inward pull is -gx.
        g_rad = -gx

        if g_rad < 0:
            negative_g_count += 1

        g_bar.append(g_rad)

        
        # Kernel softening is separate from gravity softening.
        d_kernel = np.sqrt(d2_phys + eps_kernel**2)

        K = 1.0 / d_kernel**embedding_power

        nx = dxp / d_kernel
        ny = dyp / d_kernel
        nz = dzp / d_kernel

        T = np.sum(G * m * K)

        Ax = np.sum(G * m * K * nx)
        Ay = np.sum(G * m * K * ny)
        Az = np.sum(G * m * K * nz)

        A = np.sqrt(Ax**2 + Ay**2 + Az**2)

        T_vals.append(T)
        A_vals.append(A)
        Ax_vals.append(Ax)
        Ay_vals.append(Ay)
        Az_vals.append(Az)

    g_bar = np.asarray(g_bar)

    v_bar = np.sqrt(np.maximum(r_samples * g_bar, 0.0))
    
    out = {
        "r": np.asarray(r_samples),
        "g_bar": g_bar, # (km2/s2).(1/kpc)
        "v_bar": v_bar,
        #"eps_grav": eps_grav,
        #"eps_kernel": eps_kernel,
        #"negative_g_count": negative_g_count,
    }

    # converting to m/s^2
    T_vals = np.asarray(T_vals)*KPC_KMS2_TO_MS2
    A_vals = np.asarray(A_vals)*KPC_KMS2_TO_MS2

    Ax_vals = np.asarray(Ax_vals)*KPC_KMS2_TO_MS2
    Ay_vals = np.asarray(Ay_vals)*KPC_KMS2_TO_MS2
    Az_vals = np.asarray(Az_vals)*KPC_KMS2_TO_MS2
    
    out.update({
        "T": T_vals,
        "A": A_vals,
        "I": T_vals - A_vals,
        "Ax": np.asarray(Ax_vals),
        "Ay": np.asarray(Ay_vals),
        "Az": np.asarray(Az_vals),
    })

    return out

def combine_profiles(component_profiles):
    r = component_profiles[0]["r"]

    g_bar = np.zeros_like(r, dtype=float) 
    T = np.zeros_like(r, dtype=float)
    A = np.zeros_like(r, dtype=float)
    Ax = np.zeros_like(r, dtype=float)
    Ay = np.zeros_like(r, dtype=float)
    Az = np.zeros_like(r, dtype=float)
    negative_g_count = 0

    for p in component_profiles:
        g_bar += p["g_bar"]

        # T is linear.
        T += p["T"]

        # A is not linear, but its vector components are.
        Ax += p["Ax"]
        Ay += p["Ay"]
        Az += p["Az"]

        negative_g_count += p.get("negative_g_count", 0)

    A = np.sqrt(Ax**2 + Ay**2 + Az**2)

    g_bar_m_s2 = g_bar*KPC_KMS2_TO_MS2

    return {
        "r": r,
        "g_bar": g_bar_m_s2, # (km2/s2).(1/kpc)
        "v_bar": np.sqrt(np.maximum(r * g_bar, 0.0)),
        "T": T,
        "A": A,
        "I": T - A,
        "Ax": Ax,
        "Ay": Ay,
        "Az": Az,
        "negative_g_count": negative_g_count,
    }

def get_galaxy_profile(profiles_path, galaxy, galaxy_name, ml_disk, ml_bulge, use_cached_profile):

    galaxy_profile_dir = os.path.join(profiles_path, f"disk_{ml_disk}_bulge_{ml_bulge}") 
    
    file_name = f"{galaxy_name}.pickle"
    galaxy_profile_path = os.path.join(galaxy_profile_dir, file_name)
        
    if use_cached_profile and os.path.exists(galaxy_profile_path):
        #print("Loading precomputed profile")
        with open(galaxy_profile_path, 'rb') as f:
            profiles = pickle.load(f)

    else:
        #print("Computing and caching new profiles")
        
        galaxy_df = galaxy["data"]
        #plot_surface_brightness(galaxy)
        elements, r_samples, eps_info = build_sparc_elements_v2(
            galaxy_df,
            ml_disk=ml_disk,
            ml_bulge=ml_bulge,
            include_disk=True,
            include_gas=True,
            include_bulge=True,
            gas_smoothing_passes=2,
            disk_smooth_strength=0.5,
            gas_smooth_strength=2.0,
            disk_prior_strength=0.05,
            target_arc_kpc=0.08,
        )

        individual_profiles = []

        if elements["disk"]:
            disk_prof = compute_profiles_from_elements(
            elements["disk"],
            r_samples,
            eps_info["disk"],
            embedding_power=2.0,
            G=G_KPC,
            )
            individual_profiles.append(disk_prof)

        if elements["gas"]:
            gas_prof = compute_profiles_from_elements(
            elements["gas"],
            r_samples,
            eps_info["gas"],
            embedding_power=2.0,
            G=G_KPC,
            )
            individual_profiles.append(gas_prof)

        if elements["bulge"]:
            bulge_prof = compute_profiles_from_elements(
            elements["bulge"],
            r_samples,
            eps_info["bulge"],
            embedding_power=2.0,
            G=G_KPC,
            )
            individual_profiles.append(bulge_prof)

        if len(individual_profiles) == 0:
            return None
        else:   
            profiles = combine_profiles(individual_profiles)

            #profiles = disk_prof 
        
        #galaxy_profiles["diagnostics"] = diagnostics
        #galaxy_profiles["eps_info"] = eps_info

        Path(galaxy_profile_dir).mkdir(parents=True, exist_ok=True)

        with open(galaxy_profile_path, 'wb') as f:
            pickle.dump(profiles, f)   

    return profiles