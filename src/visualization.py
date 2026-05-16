import numpy as np
import matplotlib.pyplot as plt
import shutil
import os 

from src.metrics import calculate_shape_metric, get_velocity_relation, get_threshold_window, get_valid_galaxies

from src.profile_generator import get_galaxy_profile
from src.constants import *

def generate_markdown_notes(target_path, output_file, file_format):
    """
    Scans a directory for PNG files and creates a Markdown document 
    embedding the images with space for custom notes.
    """
    # Open the markdown file for writing

    if os.path.exists(output_file):
        os.remove(output_file)
        print("Existing notes deleted")

    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Image Notes\n\n")
        
        for root_dir, sub_dirs, files in os.walk(target_path):
            # Filter the list to only include PNG / PDF files
            image_files = [f for f in files if f.lower().endswith('.' + file_format)]
            
            # Only write a directory header if there are actually image files inside it
            if image_files:
                # Get the name of the current folder for the header
                dir_name = os.path.basename(root_dir)
                if not dir_name:  # Fallback if it's the root directory
                    dir_name = "Main Directory"
                
                md_file.write(f"## Folder: {dir_name}\n\n")
                
                for file_name in image_files:
                    # Calculate the relative path so the links don't break 
                    # if you move the whole folder to another computer
                    full_path = os.path.join(root_dir, file_name)
                    rel_path = os.path.relpath(full_path, start=target_path)
                    
                    # Markdown image links require forward slashes, even on Windows
                    rel_path = rel_path.replace('\\', '/')
                    
                    # Write the image title, the embedded image, and note placeholders
                    md_file.write(f"### {file_name}\n")
                    md_file.write(f"![{file_name}]({rel_path})\n\n")
                    md_file.write("> **Notes:** \n")
                    md_file.write(f"{dir_name}\n")
                    md_file.write("> \n\n")
                    md_file.write("---\n\n") # Adds a horizontal dividing line

def get_order_bucket(threshold_radii):
    """
    Return a string that summarizes the thresholds for a given galaxy in radial order
    """
    radii = (threshold_radii["X"], threshold_radii["I"], threshold_radii["A"], threshold_radii["T"], threshold_radii["B"])
    initials = ["X", "I", "A", "T", "B"]

    # Sort column names based on the values in the row (descending)
    # argsort gives indices of sorted values; we flip it for High > Low
    
    initials = [element for radius, element in zip(radii, initials) if radius]
    radii = np.asarray([radius for radius in radii if radius])
    sorted_indices = np.argsort(radii)

    sorted_initials = np.asarray([initials[i] for i in sorted_indices])
    return "".join(sorted_initials.astype(str))

def plot_threshold_lines(ax, radii):
    """
    Plot helper that draws vertical lines representing given thresholds
    """

    X_transition_radius = radii["X"]
    I_transition_radius = radii["I"]
    A_transition_radius = radii["A"]
    T_transition_radius = radii["T"]
    B_transition_radius = radii["B"]
    
    if I_transition_radius:
        ax.axvline(x = I_transition_radius, linestyle=':', color = "red", label="$I=BG$")

    if A_transition_radius:
        ax.axvline(x = A_transition_radius, linestyle=':', color = "blue", label="$A=BG$")

    if T_transition_radius:
        ax.axvline(x = T_transition_radius, linestyle='-.', color = "green", label="$T=BG$")
    
    if X_transition_radius:
        ax.axvline(x = X_transition_radius, linestyle=':', label="$A=I$")

    if B_transition_radius:
        ax.axvline(x = B_transition_radius, linestyle='--', label=r"$I\approx0$")

def plot_fits(ax, shape_metrics):
    
    slope_array, k_array, left_radius_segment, right_radius_segment = shape_metrics

    if slope_array and k_array:

        left_slope, right_slope = slope_array

        left_k, right_k, = k_array

        left_relation_fit = left_slope*left_radius_segment + left_k

        right_relation_fit = right_slope*right_radius_segment + right_k
        
        ax.plot(left_radius_segment, left_relation_fit, color="#1f77b4", 
                lw=2.5, linestyle='--', label="Metric Fit")
        
        ax.plot(right_radius_segment, right_relation_fit, color="#1f77b4", 
                lw=2.5, linestyle='--') # Dashed to distinguish further

def plot_thresholded_curves(threshold_initial, ax, galaxy_df, threshold_window_set, relation_type, legend_on = False):
    
    labels = {
            "I": r"$\mathrm{I=BG}$", "A": r"$\mathrm{A=BG}$", 
            "T": r"$\mathrm{T=BG}$", "X": r"$\mathrm{A=I}$", "B": r"$\mathrm{I\approx0}$", 
            "G": r"$\mathrm{a_{\rm bar}=B}$",
            '25_Rmax': r"$\mathrm{R_{0.25}}$", '50_Rmax': r"$\mathrm{R_{0.5}}$", '75_Rmax': r"$\mathrm{R_{0.75}}$"}
    
    threshold_label = labels[threshold_initial]

    threshold_window = threshold_window_set[threshold_initial]
                                                  
    if threshold_window:

        threshold_radius = threshold_window["radius"]
        left_window_radius = threshold_window["left_radius"]
        right_window_radius = threshold_window["right_radius"]

        shape_metrics = calculate_shape_metric(threshold_window, galaxy_df, relation_type)
        slope_array, k_array, _, _ = shape_metrics

        if slope_array and k_array:

            sparc_radius = galaxy_df["Rad"]
            sparc_v_baryonic = galaxy_df["Vbar"]
            sparc_v_observed = galaxy_df["Vobs"]

            sparc_relation = get_velocity_relation(sparc_v_baryonic, sparc_v_observed, sparc_radius, relation_type)
            
            ax.scatter(sparc_radius, sparc_relation, marker='o', 
                color="#29ad32", alpha=0.5, edgecolors='none', label=r"$\mathrm{V_{\rm bar}^2/V_{\rm obs}^2}$")
            
            plot_fits(ax, shape_metrics)

            ax.axvspan(left_window_radius, right_window_radius, 
                color='gray', alpha=0.15, label='Fit Window', zorder=1)
            
            ax.axvline(x=threshold_radius, color='black', linestyle='--', 
                linewidth=1.2, alpha=0.6, label='Threshold', zorder=2)
            
            #ax.set_ylim(bottom=0)

            ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray', zorder=0)

            ax.set_title(f"{threshold_label}")

            if legend_on:
                # 5. Grouped Legend (Using the spacer technique)
                handles, labels = ax.get_legend_handles_labels()
                # Reorder to group: [Baryonic Data, Baryonic Fit, Spacer, Observed Data, Observed Fit]
                order = [0, 2, 1, 3] 
                # Note: You can add an empty proxy here if you want a physical gap

                ax.legend([handles[i] for i in order], [labels[i] for i in order], 
                    frameon=True, loc='best')
        else:
            ax.set_title(f"{threshold_label} fit ill conditioned")
    else:
        ax.set_title(f"{threshold_label} not observed")

def plot_sparc_curves(ax, galaxy_df, r_samples, v_bar_simulated):

    sparc_radius = galaxy_df["Rad"]
    sparc_v_baryonic = galaxy_df["Vbar"]
    sparc_v_observed = galaxy_df["Vobs"]
    sparc_v_disk = galaxy_df["Vdisk"]
    sparc_v_bulge = galaxy_df["Vbul"]
    sparc_v_gas = galaxy_df["Vgas"]
    sparc_v_disk_bulge = np.sqrt(sparc_v_disk**2 + sparc_v_bulge**2)

    ax.scatter(sparc_radius, sparc_v_observed, color="black", lw=2, label="Vobs")

    ax.scatter(sparc_radius, sparc_v_baryonic, color="blue", lw=2, label="Vbar")

    ax.plot(sparc_radius, sparc_v_disk_bulge, "--", color="red", lw = 2, label="V disk + bulge")

    ax.plot(sparc_radius, sparc_v_disk, "--", color="green", lw = 2, label="V disk")

    ax.plot(sparc_radius, sparc_v_bulge, "--", color="blue", lw = 2, label="V bulge")

    ax.plot(sparc_radius, sparc_v_gas, "--", color="red", lw = 2, label="V gas")

    ax.plot(r_samples, v_bar_simulated, color="lightgreen", lw = 2, label="Vbar simulated")

    ax.set_ylim(bottom=0)
    ax.set_title(f"")
    ax.set_xlabel("Radius (kpc)")
    #ax.set_ylabel("Velocity (km/s)")
    ax.legend(bbox_to_anchor=(1, 1), fontsize='x-small')

def plot_proxies(CONFIG, galaxy, galaxy_name, galaxy_profiles):

    local_window = CONFIG["local_window"]
    save_plot_image = CONFIG["save_plots"]
    use_bucket = CONFIG["use_buckets"]
    background_magnitude =CONFIG["plot_BG_magnitude"] # m/s2

    A_I_ratio = CONFIG["A_I_ratio"]

    image_dir = CONFIG["save_plots_path"]

    relation_type = CONFIG["relation_type"]

    threshold_window_set = {}
    
    for threshold_type in FIELD_THRESHOLDS + CONTROL_THRESHOLDS:
        
        threshold_window_set[threshold_type] = get_threshold_window(threshold_type, local_window, galaxy_profiles, background_magnitude, A_I_ratio)

    threshold_radii = {}
    for key in threshold_window_set:
        value = threshold_window_set[key]
        if value:
            threshold_radii[key] = threshold_window_set[key]["radius"]
        else:
            threshold_radii[key] = None

    order_bucket = get_order_bucket(threshold_radii)

    #print(order_bucket)

    galaxy_df = galaxy["data"]

    if use_bucket:
        image_save_dir = os.path.join(image_dir, order_bucket)
    else:
        image_save_dir = image_dir

    image_save_path = os.path.join(image_save_dir, galaxy_name + "." + CONFIG["file_format"])
    os.makedirs(image_save_dir, exist_ok=True)

    if CONFIG["plot_mode"] == "full":
        plot_full(galaxy_df, galaxy_profiles, threshold_window_set, background_magnitude, threshold_radii, relation_type)
    elif CONFIG["plot_mode"] == "compact":
        plot_compact(galaxy_df, threshold_window_set, relation_type)
    elif CONFIG["plot_mode"] == "velocity":
        plot_velocity(galaxy_df, galaxy_profiles)
    else:
        print("Unknown plotting mode")
    
    if save_plot_image:
        plt.savefig(image_save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    
    if CONFIG["plot_mode"] == "compact":
        _, ax_threshold = plt.subplots(figsize = (5.8, 4.0), sharex=True, sharey=True, constrained_layout=True)

        image_save_path_threshold = os.path.join(image_save_dir, galaxy_name + "_thresholds" + "." + CONFIG["file_format"])
        plot_compact_threshold(ax_threshold, galaxy_profiles, threshold_radii, background_magnitude)

        if save_plot_image:
            plt.savefig(image_save_path_threshold, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

def plot_compact(galaxy_df, threshold_window_set, relation_type):

    plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10,
    "lines.markersize": 5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,    # Despine for a modern look
    "axes.spines.right": False
    })

    fig, axes = plt.subplots(2, 3, figsize=(6, 4), constrained_layout=True)
    axs = axes.flatten()

    for i in [1, 3, 4]:
        axs[i].sharex(axs[0])
        axs[i].sharey(axs[0])

    # 3. Clean up labels (Manual sharing doesn't auto-hide labels like the global keyword does)
    # Hide X labels for the top row of the shared group
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[1].get_xticklabels(), visible=False)
    # Hide Y labels for the middle column of the shared group
    plt.setp(axs[1].get_yticklabels(), visible=False)
    plt.setp(axs[4].get_yticklabels(), visible=False)

    ax = axs[0]
    plot_thresholded_curves("A", ax, galaxy_df, threshold_window_set, relation_type)
    handles, labels = ax.get_legend_handles_labels()

    ax = axs[1]
    plot_thresholded_curves("B", ax, galaxy_df, threshold_window_set, relation_type)
    if not labels:
        handles, labels = ax.get_legend_handles_labels()

    ax = axs[3]
    plot_thresholded_curves("25_Rmax", ax, galaxy_df, threshold_window_set, relation_type)
    if not labels:
        handles, labels = ax.get_legend_handles_labels()

    ax = axs[4]
    plot_thresholded_curves("50_Rmax", ax, galaxy_df, threshold_window_set, relation_type)
    if not labels:
        handles, labels = ax.get_legend_handles_labels()

    ax = axs[5]

    sparc_radius = np.asarray(galaxy_df["Rad"])
    sparc_v_baryonic = galaxy_df["Vbar"]
    sparc_v_observed = galaxy_df["Vobs"]

    ax.plot(sparc_radius, sparc_v_baryonic, color="#1f77b4", linewidth = 2.0, label="V. Baryonic")
    
    ax.plot(sparc_radius, sparc_v_observed, color="#ff7f0e", linewidth = 2.0, label="V. Observed")

    ax.tick_params(labelleft=True, labelbottom=True)
    main_handles, main_labels = ax.get_legend_handles_labels()

    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'Velocity ($\mathrm{km\,{s}^{-1}}$)')
    ax.set_title(r'SPARC Velocity')
    
    ax = axs[2]
    ax.axis('off')
    ax.legend(handles + main_handles, labels + main_labels, loc='center', labelspacing=1.5, frameon=False)
    
    fig.supxlabel("Radius (kpc)")
    fig.supylabel(r"Dimensionless metric")

def plot_compact_threshold(ax, galaxy_profiles, threshold_radii, background):

    T_galaxy = galaxy_profiles["T"]
    A_vals = galaxy_profiles["A"]
    I_galaxy = galaxy_profiles["I"]
    r_samples = galaxy_profiles["r"]


    ax.plot(r_samples, T_galaxy, color="#2ca02c", lw=2, label="T")
    ax.plot(r_samples, I_galaxy, color="#d62728", lw=2, label="I")
    ax.plot(r_samples, A_vals, color="#1f77b4", lw=2, label="A")
    
    ax.plot([0], [0], color='none', label = " ") # Legend spacing

    ax.axhline(y = background, linestyle='-.', label="BG")

    A_transition_radius = threshold_radii["A"]
    B_transition_radius = threshold_radii["B"]

    if A_transition_radius:
        ax.axvline(x = A_transition_radius, linestyle=':', label=r"$A=BG$")

    if B_transition_radius:
        ax.axvline(x = B_transition_radius, linestyle='--', label=r"$I\approx0$")

    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel(r"$\mathrm{m\,s^{-2}}$")

    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(9)

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)

def plot_velocity(galaxy_df, galaxy_profiles):

    r_samples = galaxy_profiles["r"]
    v_bar_simulated = galaxy_profiles["v_bar"]

    fig, ax = plt.subplots()
    plot_sparc_curves(ax, galaxy_df, r_samples, v_bar_simulated)

def plot_full(galaxy_df, galaxy_profiles, threshold_window_set, background_magnitude, threshold_radii, relation_type):

    g_bar = galaxy_profiles["g_bar"]

    T_galaxy = galaxy_profiles["T"]
    A_vals = galaxy_profiles["A"]
    I_galaxy = galaxy_profiles["I"]
    r_samples = galaxy_profiles["r"]
    v_bar_simulated = galaxy_profiles["v_bar"]

    BG_val = background_magnitude
    
    plt.figure(figsize=(10, 10)) 

    plot_dimensions = (4, 3)

    ax = plt.subplot2grid(plot_dimensions, (0, 0))
    
    plot_thresholded_curves("I", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (0, 1))

    plot_thresholded_curves("B", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (0, 2), rowspan = 2)
    
    ax.plot(r_samples, A_vals, color="blue", label = "A_vals")

    ax.plot(r_samples, g_bar, linestyle='--', color="black", label = "a_bar")

    ax.plot(r_samples, T_galaxy, color="lightgreen", label = "T_galaxy")
    ax.plot(r_samples, I_galaxy, color="red", label = "I_galaxy")
    ax.axhline(y = BG_val, linestyle='-', label="B inf")

    plot_threshold_lines(ax, threshold_radii)

    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel(r"$\mathrm{m\,s^{-2}}$")
    
    ax.legend()

    ax = plt.subplot2grid(plot_dimensions, (1, 0))
    plot_thresholded_curves("A", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (1, 1))
    plot_thresholded_curves("X", ax, galaxy_df, threshold_window_set, relation_type)
    
    ax = plt.subplot2grid(plot_dimensions, (2, 0))
    plot_thresholded_curves("G", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (2, 1))
    plot_thresholded_curves("T", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (2, 2))

    plot_sparc_curves(ax, galaxy_df, r_samples, v_bar_simulated)

    ax = plt.subplot2grid(plot_dimensions, (3, 0))
    plot_thresholded_curves("25_Rmax", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (3, 1))
    plot_thresholded_curves("50_Rmax", ax, galaxy_df, threshold_window_set, relation_type)

    ax = plt.subplot2grid(plot_dimensions, (3, 2))
    plot_thresholded_curves("75_Rmax", ax, galaxy_df, threshold_window_set, relation_type)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.8)

def make_plots(CONFIG):
    
    galaxy_data_parsed = get_valid_galaxies(CONFIG, CONFIG["plot_M_L_DISK"], CONFIG["plot_M_L_BULGE"])
        
    if len(galaxy_data_parsed) == 0:
        print(f"No valid galaxies for gas cutoff threshold {CONFIG["gas_proportion_cutoff"]}")
        return
    
    galaxy_name_array = list(galaxy_data_parsed)
    
    use_cached_profiles = CONFIG["use_cached_profiles"]

    ml_disk = CONFIG["plot_M_L_DISK"]
    ml_bulge = CONFIG["plot_M_L_BULGE"]
    
    save_plot = CONFIG["save_plots"]
        
    image_dir = CONFIG["save_plots_path"]

    if save_plot:
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
            print("Existing plots deleted")

    galaxy_count = 0
    for galaxy_name in galaxy_name_array:
        galaxy = galaxy_data_parsed[galaxy_name]
        galaxy_profiles = get_galaxy_profile(CONFIG["profiles_path"], galaxy, galaxy_name, ml_disk, ml_bulge, use_cached_profiles)

        galaxy_count += 1

        if not galaxy_profiles:
            print("Empty profile, skipped")
            continue
        
        print(f"Name: {galaxy_name}, count: {galaxy_count} / {len(galaxy_name_array)}")

        plot_proxies(CONFIG, galaxy, galaxy_name, galaxy_profiles)
    
    target_folder = image_dir
    
    if save_plot:
        print(f"Scanning '{target_folder}' for images...")
        generate_markdown_notes(target_folder, CONFIG["save_markdown_path"], CONFIG["file_format"])
        print("Done!")
