# Threshold Structure in Baryonic Field Proxies and Galactic Rotation Curves

This repository contains the official Python implementation for the paper **"Threshold Structure in Baryonic Field Proxies and Galactic Rotation Curves"**.

This project develops a computational pipeline to reconstruct 3D baryonic mass distributions from galactic kinematics data, simulates localized gravitational vector fields, and evaluates empirical relationships between baryonic vector cancellation boundaries and galactic rotation-curve regimes using the SPARC dataset.

---

## Overview

Observed galactic rotation curves consistently deviate from Newtonian predictions based on visible matter. Rather than evaluating standard net baryonic acceleration alone, this project implements a phenomenological approach to explore whether additional spatial information is encoded within the structural cancellation of the baryonic field itself.

We forward-model annular disks, gas profiles, and spherical bulges into discrete point-mass elements to extract three acceleration-like scalar field proxies at a test point at radius $r$:

* **Total Scalar Proxy ($T$):** The scalar sum of inverse-square Newtonian contributions prior to vector cancellation.
* **Net Anisotropic Proxy ($A$):** The magnitude of the net Newtonian baryonic acceleration vector field.
* **Residual Isotropic Proxy ($I = T - A$):** A spatial descriptor quantifying the local balance of directionally canceling baryonic support.

By identifying radius thresholds where these proxies cross one another or cross an empirical reference scale ($B_G = 1.2 \times 10^{-10} \text{ m s}^{-2}$), the pipeline assesses systematic variations in local trends of the dimensionless acceleration-ratio metric $Q_g = V_{bar}^2/V_{obs}^2$.

---

## Repository Structure

```text
baryonic-field-thresholds/
├── config/
│   └── config.yaml              # Centralized runtime and experiment configuration
├── src/
│   ├── __init__.py
│   ├── constants.py             # Physical constants and unit conversion scales
│   ├── data_loader.py           # SPARC archive streaming and M/L scaling math
│   ├── element_builder.py       # Non-negative mass fitting and 3D point-mass mesh generation
│   ├── metrics.py               # Spline cross-detection and Wilcoxon signed-rank tests
│   ├── profile_generator.py     # Gravity & kernel-softened field evaluation loop
│   ├── tables.py                # Parse statistical results into print and latex tables
│   └── visualization.py         # Multi-panel kinematics and proxy plotting layouts
├── main.py                      # Unified pipeline entry point
├── requirements.txt             # Environment dependencies
└── README.md

```

---

## System Requirements & Installation

This codebase requires **Python 3.12+**. Clone the repository and install the core scientific computing dependencies:

```bash
git clone https://github.com/yfikret/baryonic-field-thresholds.git
cd baryonic-field-thresholds
pip install -r requirements.txt

```

### `requirements.txt` Dependencies

```text
matplotlib==3.10.8
numpy==2.4.4
pandas==3.0.2
PyYAML==6.0.3
scipy==1.17.1
```

---

## Data Setup

The repository utilizes the mass models from the **Spitzer Photometry and Accurate Rotation Curves (SPARC)** database.

1. Download the Newtonian Mass Models archive (`Rotmod_LTG.zip`) directly from the [SPARC Database](https://astroweb.case.edu/SPARC/). 
Direct download link: https://astroweb.case.edu/SPARC/Rotmod_LTG.zip.
2. Create a folder named `sparc_archives/` at the repository root.
3. Place the unmodified `Rotmod_LTG.zip` file inside that folder:
```text
baryonic-field-thresholds/sparc_archives/Rotmod_LTG.zip

```


---

## Verification & Usage

The entire experimental pipeline is managed using `main.py` and configured via `config/config.yaml`.

### 1. Running Statistical Experiments

To calculate threshold crossings, fit local linear trends to the velocity metrics, and output statistical summaries across mass-to-light ratio combinations:

1. Open `config/config.yaml` and verify the mode is set to statistics:
```yaml
mode: "run_stats"
table_type: "print"  # Options: "print", "latex_full", "latex_compact"

```


2. Execute the entry script:
```bash
python main.py

```



The pipeline dynamically determines the number of active test evaluations, runs Wilcoxon signed-rank tests, and outputs table matrix indicators with a calculated Bonferroni-corrected alpha boundary.

### 2. Generating Kinematics and Proxy Profiles

To visually inspect mass element profiles, calculated proxy curves, and windowed local linear trend lines for individual galaxies:

1. Update `config/config.yaml` to target visualization generation:
```yaml
mode: "generate_plots"
plot_mode: "full"   # Options: "full", "compact", "velocity"
save_plots: True

```


2. Execute the entry script:
```bash
python main.py

```



Plots will be systematically rendered and saved under `./plots/images/`. An automated markdown compilation script will generate an updated `image_notes.md` notebook alongside the directory structure for easy, scannable analysis.

---

## Core Pipeline Architecture

* **`element_builder.py`:** Builds mass density maps from SPARC velocity and surface brightness data.
* **`profile_generator.py`:** Generates the proxies A, I, and T.
* **`metrics.py`:** Contains functions to calculate thresholds and applies statistical tests.

---

## Citation

If you make use of this implementation or the associated baryonic field proxy methodology in your research, please cite the preprint:

```tex
@techreport{yalcinbas2026threshold,
  title={Threshold Structure in Baryonic Field Proxies and Galactic Rotation Curves},
  author={Yalcinbas, M. Fikret},
  institution={GitHub Repository},
  year={2026},
  type={Technical Report},
  url={https://github.com/yfikret/baryonic-field-thresholds}
}
```

## Acknowledgements

* The author acknowledges the use of automated tools for coding assistance, documentation drafting, and structural optimization during the development of this repository.
* Dynamic kinematics data provided courtesy of the SPARC (Spitzer Photometry and Accurate Rotation Curves) database.