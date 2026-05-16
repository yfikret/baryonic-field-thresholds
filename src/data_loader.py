
import zipfile
import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

ROT_COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]

def apply_m_l(parsed_data_path, m_l_disk, m_l_bulge):
    
    with open(parsed_data_path, 'rb') as f:
        galaxy_data_parsed = pickle.load(f)

    for name in galaxy_data_parsed:
        df = galaxy_data_parsed[name]["data"]

        #weighted_sum = df["Vgas"] * np.abs(df["Vgas"]) + m_l_disk * df["Vdisk"] * np.abs(df["Vdisk"]) + m_l_bulge * df["Vbul"] * np.abs(df["Vbul"])

        weighted_sum = df["Vgas"] * np.abs(df["Vgas"]) + m_l_disk * np.square(df["Vdisk"]) + m_l_bulge * np.square(df["Vbul"])
        
        df["Vbar"] = np.sqrt(np.maximum(weighted_sum, 0.0))
        
        df["Vdisk_unity"] = df["Vdisk"].copy()
        df["Vbul_unity"] = df["Vbul"].copy()

        df["Vdisk"] = np.sqrt(m_l_disk)* df["Vdisk"]
        df["Vbul"] = np.sqrt(m_l_bulge)* df["Vbul"]

        galaxy_data_parsed[name]["data"] = df

    return galaxy_data_parsed


def parse_galaxy_data(archive_path, parsed_data_path):
    galaxy_data_parsed = load_rotmod_zip(archive_path)

    with open(parsed_data_path, 'wb') as f:
        pickle.dump(galaxy_data_parsed, f)  

def parse_rotmod_text(text, galaxy_name=""):
    lines = text.splitlines()

    distance_mpc = None
    for line in lines:
        if not line.startswith("#"):
            break
        m = re.search(r"Distance\s*=\s*([0-9.]+)\s*Mpc", line)
        if m:
            distance_mpc = float(m.group(1))

    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=ROT_COLS,
    )

    return {
        "galaxy": galaxy_name,
        "distance_mpc": distance_mpc,
        "data": df,
    }

def load_rotmod_zip(zip_path):
    galaxies = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith("_rotmod.dat"):
                continue
            text = z.read(name).decode("utf-8", errors="replace")
            galaxy = name.replace("_rotmod.dat", "")
            galaxies[galaxy] = parse_rotmod_text(text, galaxy_name=galaxy)
    return galaxies
