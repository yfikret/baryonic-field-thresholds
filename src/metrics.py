import os
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

from src.tables import generate_big_table, generate_compact_table

from src.profile_generator import get_galaxy_profile
from src.data_loader import parse_galaxy_data, apply_m_l
from src.constants import *


def get_gas_component_proxy(galaxy_df):
    sparc_v_disk = galaxy_df["Vdisk"]
    sparc_v_bulge = galaxy_df["Vbul"]
    sparc_v_gas = galaxy_df["Vgas"]

    gas_component_proxy = np.mean(sparc_v_gas**2 / (sparc_v_disk**2 + sparc_v_bulge**2 + sparc_v_gas**2))
    return gas_component_proxy

def apply_gas_cutoff(gas_threshold, galaxy_data_parsed, galaxy_data_fiduciary):

    filtered_galaxies = {}

    for galaxy_name in galaxy_data_parsed:

        galaxy_fiduciary = galaxy_data_fiduciary[galaxy_name]["data"]

        gas_velocity_content = get_gas_component_proxy(galaxy_fiduciary) 
        
        if gas_velocity_content <= gas_threshold:
            filtered_galaxies[galaxy_name] = galaxy_data_parsed[galaxy_name]

    return filtered_galaxies

def get_statistical_metadata(CONFIG):
    
    threshold_selection = CONFIG["threshold_types"] 
    background_magnitude_array = CONFIG["stats_BG"]

    gas_cutoff_count = CONFIG["total_gas_cutoffs"]
    ml_pair_count = len(CONFIG["stats_M_L_DISK_BULGE_pairs"])

    bg_threshold_types_count = len(BG_FIELD_THRESHOLDS) 
    non_bg_threshold_types_count = len(NON_BG_FIELD_THRESHOLDS) + len(CONTROL_THRESHOLDS)
    background_value_count = len(background_magnitude_array)

    base_p_value = 0.05
    total_tests = gas_cutoff_count*ml_pair_count*(background_value_count*bg_threshold_types_count + non_bg_threshold_types_count)

    bonferroni_p_value = base_p_value / total_tests
    
    metadata = {"total_tests": total_tests, "bonferroni_p_value": bonferroni_p_value, 
                "threshold_selection": threshold_selection}
    
    return metadata

def get_valid_galaxies(CONFIG, ml_disk, ml_bulge):
    parsed_data_path = CONFIG["SPARC_parsed_path"]
    
    if not os.path.exists(parsed_data_path):
        print("Parsing SPARC data")
        parse_galaxy_data(CONFIG["SPARC_archive_path"], parsed_data_path)

    sparc_data_parsed = apply_m_l(parsed_data_path, ml_disk, ml_bulge)

    sparc_data_fiduciary = apply_m_l(parsed_data_path, ML_DISK_FIDUCIARY, ML_BULGE_FIDUCIARY)

    if CONFIG["sample_galaxies"]:
        sample_galaxies = CONFIG["sample_galaxies"]

        sparc_data_parsed = {name: sparc_data_parsed[name] for name in sample_galaxies if name in sparc_data_parsed}

    total_galaxies = len(sparc_data_parsed)
    
    gas_cutoff = CONFIG["gas_proportion_cutoff"]
    
    sparc_data_filtered = apply_gas_cutoff(gas_cutoff, sparc_data_parsed, sparc_data_fiduciary)
    
    total_valid_galaxies = len(sparc_data_filtered)
    
    print(f"Gas threshold cutoff: {gas_cutoff}")
    print(f"Total entries used: {total_valid_galaxies}/{total_galaxies}")
    print("")

    return sparc_data_filtered


def get_threshold_window(threshold_type, local_window_proportion, galaxy_profiles, background_magnitude, A_I_ratio):
    """
    Calculate the radial fitting window for a provided threshold type. Return None if crossing does not exist.
    """
    r_samples = galaxy_profiles["r"]

    r_max = r_samples[-1]
    r_min = r_samples[0]
    local_window_length = r_max*local_window_proportion

    BG_val = background_magnitude
    
    if threshold_type in FIELD_THRESHOLDS:
        if threshold_type == "X":
            difference_array = galaxy_profiles["I"] - galaxy_profiles["A"]
        elif threshold_type == "B":
            difference_array = A_I_ratio*galaxy_profiles["I"] - galaxy_profiles["A"]
        elif threshold_type == "G":
            difference_array = galaxy_profiles["g_bar"] - BG_val
        else:
            difference_array = galaxy_profiles[threshold_type] - BG_val

        transition_radius = find_crossing(r_samples, difference_array)

        if transition_radius:
            left_window_radius = np.max(((transition_radius - local_window_length), r_min))
            right_window_radius = np.min(((transition_radius + local_window_length), r_max))
        
            metrics = {
                "radius": transition_radius, "left_radius": left_window_radius, "right_radius": right_window_radius
                }
        else:   
            metrics = None

    elif threshold_type in CONTROL_THRESHOLDS:
        
        if threshold_type == "50_Rmax":
            control_radius = r_max*0.5
        elif threshold_type == "25_Rmax":
            control_radius = r_max*0.25
        elif threshold_type == "75_Rmax":
            control_radius = r_max*0.75
        elif threshold_type == "V_bar_max":
            control_radius = r_samples[np.argmax(galaxy_profiles["v_bar"])]
        
        left_window_radius = np.max(((control_radius - local_window_length), r_min))
        right_window_radius = np.min(((control_radius + local_window_length), r_max))

        metrics = {"radius": control_radius, "left_radius": left_window_radius, "right_radius": right_window_radius}   

    return metrics

def get_thresholded_arrays(CONFIG, local_window, threshold_type, galaxy_data_parsed, background_magnitude, ml_disk, ml_bulge):
    
    A_I_ratio = CONFIG["A_I_ratio"]
    relation_type = CONFIG["relation_type"]
    use_cached_profiles = CONFIG["use_cached_profiles"]

    #BG_val = background_magnitude

    left_relation_array = []
    right_relation_array = []

    for galaxy_name in galaxy_data_parsed:
        galaxy = galaxy_data_parsed[galaxy_name]
        
        galaxy_profiles = get_galaxy_profile(CONFIG["profiles_path"], galaxy, galaxy_name, ml_disk, ml_bulge, use_cached_profiles)
        
        threshold_window = get_threshold_window(threshold_type, local_window, galaxy_profiles, background_magnitude, A_I_ratio)
        
        df = galaxy["data"]

        slope_array, _, _, _ = calculate_shape_metric(threshold_window, df, relation_type)
        
        if slope_array:
            left_relation, right_relation = slope_array

            left_relation_array.append(left_relation)
            right_relation_array.append(right_relation)
    

    left_relation_array = np.asarray(left_relation_array)
    right_relation_array = np.asarray(right_relation_array)

    return left_relation_array, right_relation_array

def get_velocity_relation(vbar, vobs, radii, relation_type):
    """
    Calculate the metric that relates observed and baryonic velocity for regime change assessment
    """
    if relation_type == "square_difference_over_r":
        relation = (np.square(vbar) - np.square(vobs)) / radii
    elif relation_type == "squared_ratio":
        relation = np.square(vbar) / np.square(vobs)
    elif relation_type == "ratio":
        relation = vbar / vobs
        
    return relation

def find_crossing(radius_array, difference_array):
    """
    Use spline interpolation to find the crossing point radius for a provided difference array
    """

    difference_spline = UnivariateSpline(radius_array, difference_array, s=0)
    crossing_points = difference_spline.roots()
    
    if len(crossing_points) > 0:
        transition_radius = crossing_points[-1]
    else:
        transition_radius = None

    return transition_radius

def calculate_shape_metric(threshold_window, galaxy_df, relation_type):
    """
    For a given threshold, calculate the linear fit to the chosen metric. Return slope and intercept for each half-window.
    """

    min_fit_points = 3
    min_radius_proportion = 0.05

    sparc_radius = np.asarray(galaxy_df["Rad"])
    sparc_v_baryonic = np.asarray(galaxy_df["Vbar"])
    sparc_v_observed = np.asarray(galaxy_df["Vobs"])
    
    r_max = sparc_radius[-1]

    min_window_range = r_max*min_radius_proportion

    if not threshold_window:
        return None, None, None, None
    
    transition_radius = threshold_window["radius"]
    
    threshold_index = np.argmax((sparc_radius - transition_radius) > 0)
    sparc_transition_radius = sparc_radius[threshold_index]
    
    left_index = np.where((sparc_radius - threshold_window["left_radius"]) >= 0)[0][0]

    right_index = np.where((sparc_radius - threshold_window["right_radius"]) <= 0)[0][-1]

    if sparc_transition_radius < transition_radius:
        left_index_range = slice(left_index, threshold_index + 1)
        right_index_range = slice(threshold_index + 1, right_index + 1)
    else:
        left_index_range = slice(left_index, threshold_index)
        right_index_range = slice(threshold_index, right_index + 1)

    # Fit slope to each four sections
    left_radius_array = sparc_radius[left_index_range]
    right_radius_array = sparc_radius[right_index_range]

    if left_radius_array.size < min_fit_points or right_radius_array.size < min_fit_points:
        return None, None, None, None
    
    left_window_range = transition_radius - left_radius_array[0]
    right_window_range = right_radius_array[-1] - transition_radius

    if left_window_range < min_window_range or right_window_range < min_window_range:
        return None, None, None, None
    
    left_baryonic = sparc_v_baryonic[left_index_range]
    left_observed = sparc_v_observed[left_index_range]
    left_relation = get_velocity_relation(left_baryonic, left_observed, left_radius_array, relation_type)
    
    left_slope, left_k = np.polyfit(left_radius_array, left_relation, 1)

    right_baryonic = sparc_v_baryonic[right_index_range]
    right_observed = sparc_v_observed[right_index_range]
    right_relation = get_velocity_relation(right_baryonic, right_observed, right_radius_array, relation_type)

    right_slope, right_k = np.polyfit(right_radius_array, right_relation, 1)

    slope_array = (left_slope, right_slope)
    k_array = (left_k, right_k)
    
    return slope_array, k_array, left_radius_array, right_radius_array 

def get_results(CONFIG, galaxy_data_parsed, ml_disk, ml_bulge):

    """
    Calculate velocity-based metrics across thresholds and assess Bonferroni-corrected significance
    """

    background_magnitude_array = CONFIG["stats_BG"]

    threshold_selection = CONFIG["threshold_types"] 

    local_window = CONFIG["local_window"] 

    results = []
    
    for index, background_magnitude in enumerate(background_magnitude_array):
        
        if threshold_selection == "all":
            
            if index == len(background_magnitude_array) - 1: # Last item
                threshold_types = BG_FIELD_THRESHOLDS + NON_BG_FIELD_THRESHOLDS + CONTROL_THRESHOLDS
            else:
                threshold_types = BG_FIELD_THRESHOLDS
        
        else:
            if len(threshold_selection) == 1:
                threshold_types = (threshold_selection,)
            else:
                threshold_types = threshold_selection

        for threshold_type in threshold_types:
            left_relation_array, right_relation_array = get_thresholded_arrays(CONFIG, local_window, threshold_type, galaxy_data_parsed, background_magnitude, ml_disk, ml_bulge)
            
            delta = left_relation_array - right_relation_array
            delta_mean = np.mean(delta)

            delta_median = np.median(delta)

            delta_std = np.std(delta)

            positive_fraction = (delta > 0).mean()

            #delta_skew = stats.skew(delta)

            #print("Running Wilcoxon signed-rank test")
            # Perform the Wilcoxon signed-rank test
            # By default, it tests the null hypothesis that differences are symmetric about zero
            _, wilcoxon_p_value = stats.wilcoxon(left_relation_array, right_relation_array)

            threshold_results = {"threshold_type": threshold_type,
                                "delta": delta, "delta_mean": delta_mean, "delta_median": delta_median,
                                "delta_std": delta_std, "positive_fraction": positive_fraction,
                                "p_value": wilcoxon_p_value, "background": background_magnitude}
            
            results.append(threshold_results)
        
    return results


def run_stats(CONFIG):
    results_for_compact = []

    metadata = get_statistical_metadata(CONFIG)

    for M_L_DISK_BULGE_pair in CONFIG["stats_M_L_DISK_BULGE_pairs"]:
        
        ml_disk = M_L_DISK_BULGE_pair[0]
        ml_bulge = M_L_DISK_BULGE_pair[1]

        print(f"")
        print(f"M/L Disk: {ml_disk}")
        print(f"M/L Bulge: {ml_bulge}")

        valid_galaxies = get_valid_galaxies(CONFIG, ml_disk, ml_bulge)

        if len(valid_galaxies) == 0:
            print(f"No valid galaxies for gas cutoff threshold {CONFIG["gas_proportion_cutoff"]}")
            continue
        
        results = get_results(CONFIG, valid_galaxies, ml_disk, ml_bulge)

        if CONFIG["table_type"] == "latex_compact":
            results_for_compact.append({"results": results, "ml_disk": ml_disk,  "ml_bulge": ml_bulge})
        else:
            generate_big_table(CONFIG, results, metadata, ml_disk, ml_bulge)
        
    if CONFIG["table_type"] == "latex_compact":
        generate_compact_table(CONFIG, results_for_compact)
    
    print(f"Total tests: {metadata["total_tests"]}")
    print(f"Bonferroni corrected p-value: {metadata["bonferroni_p_value"]:.2e}")