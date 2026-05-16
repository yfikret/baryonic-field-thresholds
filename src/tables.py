import numpy as np
import os
from pathlib import Path

from src.constants import *

def get_threshold_label(background, threshold_type, format_mode):

    BG_specifier_string = f" ({background:.1e})"

    if format_mode ==  "latex_full":
        threshold_labels = {
            "I": r"$\mathrm{I=BG}$" + BG_specifier_string, "A": r"$\mathrm{A=BG}$" + BG_specifier_string, 
            "T": r"$\mathrm{T=BG}$" + BG_specifier_string, "X": r"$\mathrm{A=I}$", "B": r"$\mathrm{I\approx0}$", 
            "G": r"$\mathrm{a_{\rm bar}=BG}$" + BG_specifier_string}
            
        threshold_labels.update({   "25_Rmax": r"$\mathrm{0.25 \times R_{max}}$",
                                    "50_Rmax": r"$\mathrm{0.50 \times R_{max}}$",
                                    "75_Rmax": r"$\mathrm{0.75 \times R_{max}}$",
                                    "V_bar_max": r"$max(V_{bar})$" 
                                })
                                    
    elif format_mode == "latex_compact":

        threshold_labels = {
            "I": r"I=BG", "A": r"A=BG", 
            "T": r"T=BG", "X": r"A=I", "B": r"I\approx0", 
            "G": r"a_{\rm bar}=BG"}
        
        threshold_labels.update({   "25_Rmax": "0.25",
                                    "50_Rmax": "0.50",
                                    "75_Rmax": "0.75",
                                    "V_bar_max": r"max(v_{bar})"
                                })
    else:
        threshold_labels = {
        "I": "I=BG" + BG_specifier_string, "A": "A=BG" + BG_specifier_string, 
        "T": "T=BG" + BG_specifier_string, "X": "A=I", "B": "I ~= 0", 
        "G": "a_bar=BG" + BG_specifier_string,
        "25_Rmax": "0.25", "50_Rmax": r"0.50", "75_Rmax": "0.75", "V_bar_max": "V_bar_max"} 

    return threshold_labels[threshold_type]

def write_latex_table_head(column_labels, f):

    total_columns = len(column_labels)

    table_label = ""
    caption = ""

    column_string = " & ".join(column_labels)
    
    column_types = "l" + "c"*(total_columns - 1)

    print(r"\begin{table}[h]", file = f)
    print(r"\centering", file = f)
    print(r"\small", file = f)
    print(f"\\caption{{{caption}}}", file = f)
    print(f"\\label{{tab:{table_label}}}", file = f)
    print(f"\\begin{{tabular}}{{{column_types}}}", file = f)
    print("\\toprule", file = f)
    print(column_string, " \\\\", file = f)
    print("\\midrule", file = f)

def write_latex_table_end(f):

    print("\\bottomrule", file = f)
    print(r"\end{tabular}", file = f)
    print(r"\end{table}", file = f)


def construct_compact_threshold_string(row):

    threshold_label_string = get_threshold_label(row["background"], row["threshold_type"], format_mode = "latex_compact")

    threshold_string = r"\begin{tabular}[t]{@{}l@{}}" \
        + r"$R_{" + f"{threshold_label_string}" \
        + r"}$ ($p$: " + f"{row["p_value"]:.2e}" + r")" \
        + r"\\ \addlinespace" + r" $med.:" + f"{row["delta_median"]:.2f}" \
        + r"\quad f_p: " + f"{row["positive_fraction"]:.2f}" + r"$\end{tabular}"

    return threshold_string

def generate_compact_result_rows(results, ml_disk, ml_bulge):

    field_rows = [row for row in results if row["threshold_type"] in FIELD_THRESHOLDS]

    best_field_index = np.argmin([row["p_value"] for row in field_rows]) 

    control_rows = [row for row in results if row["threshold_type"] in CONTROL_THRESHOLDS]

    best_control_index = np.argmin([row["p_value"] for row in control_rows]) 

    best_field_row = field_rows[best_field_index]
    
    best_control_row = control_rows[best_control_index]

    field_control_margin = np.log10(best_control_row["p_value"]/best_field_row["p_value"])

    field_control_margin_string = f"{field_control_margin:.2f}"

    ml_disk_string = f"{ml_disk}"
    
    ml_bulge_string = f"{ml_bulge}"

    best_field_string = construct_compact_threshold_string(best_field_row)
    best_control_string = construct_compact_threshold_string(best_control_row)
    
    formatted_row = (
                ml_disk_string,
                ml_bulge_string,
                best_field_string,
                best_control_string,
                field_control_margin_string
            )
    
    formatted_results = (formatted_row,)

    return formatted_results

def generate_compact_table(CONFIG, results_for_compact):

    column_labels = ("M/L Disk", "M/L Bulge", "Strongest field", "Strongest control", "Margin")

    if CONFIG["latex_output_path"]:

        output_path = os.path.join(CONFIG["latex_output_path"], "compact_table.txt")
        parent = Path(output_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        f = open(output_path, "w")
    else:
        f = None

    write_latex_table_head(column_labels, f)

    for entry in results_for_compact:
        results = entry["results"]

        ml_disk = entry["ml_disk"]
        ml_bulge = entry["ml_bulge"]
        compact_result_rows = generate_compact_result_rows(results, ml_disk, ml_bulge)

        for row in compact_result_rows:
        
            value_string = " & ".join(row)
            print(value_string, " \\\\", file = f)

    write_latex_table_end(f)

    if CONFIG["latex_output_path"]:
        f.close()
    

def generate_full_result_rows(results, metadata, format_mode):
    
    threshold_selection = metadata["threshold_selection"]
    bonferroni_p_value = metadata["bonferroni_p_value"]
    #total_tests = metadata["total_tests"]

    formatted_results = []

    for row in results:
        threshold_type = row["threshold_type"]
        p_value = row["p_value"]
        positive_fraction = row["positive_fraction"]
        delta_mean = row["delta_mean"]
        delta_median = row["delta_median"]
        delta_std = row["delta_std"]
        delta = row["delta"]
        background = row["background"]

        if p_value < bonferroni_p_value:
            p_value_string = f"{p_value:.2e} (S)"
        else:
            p_value_string = f"{p_value:.2e}"

        threshold_label_string = get_threshold_label(background, threshold_type, format_mode)

        formatted = (threshold_label_string,
                    p_value_string,
                    f"{positive_fraction:.3f}",
                    f"{delta_mean:.3f}",
                    f"{delta_median:.3f}",
                    f"{delta_std:.3f}",
                    f"{len(delta)}"
                    )
        formatted_results.append(formatted)

        if threshold_selection == "all" and threshold_type in ("T", "B"):
            formatted_results.append("Midrow")
    
    return formatted_results

def write_full_latex_table(result_rows, column_labels, config_output_path, ml_disk, ml_bulge):

    if config_output_path:
        output_path = os.path.join(config_output_path, f"table_disk_{ml_disk}_bulge_{ml_bulge}.txt")
        parent = Path(output_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        f = open(output_path, "w")

    else:
        f = None
    
    write_latex_table_head(column_labels, f)

    for row in result_rows:
        if row == "Midrow":
            print("\\midrule", file = f)
        else:
            value_string = " & ".join(row)
            print(value_string, " \\\\", file = f)
    
    write_latex_table_end(f)

    if config_output_path:
        f.close()

def write_print_table(result_rows, column_labels):
    
    width = 18
    labels_aligned = "".join([f"{item:<{width}}" for item in column_labels])
    
    print(labels_aligned, "\n")

    for row in result_rows:
        if row == "Midrow":
            print("")
        else:
            row_aligned = "".join([f"{item:<{width}}" for item in row])
            print(row_aligned)
    
def generate_big_table(CONFIG, results, metadata, ml_disk, ml_bulge):

    delta_string = r"${\Delta} $"
    column_labels_full_latex = ("Threshold", "p-value", 
                                delta_string + "P. fraction", delta_string + "Mean", 
                                delta_string + "Median", delta_string + "St. deviation", "n")

    column_labels_full_print = ("Threshold", "p-value", 
                                "P. fraction", "Mean", 
                                "Median", "St. deviation", "Valid thresholds")


    full_result_rows = generate_full_result_rows(results, metadata, format_mode = CONFIG["table_type"])
    
    if CONFIG["table_type"] == "latex_full":
        write_full_latex_table(full_result_rows, column_labels_full_latex, CONFIG["latex_output_path"], ml_disk, ml_bulge)
    elif CONFIG["table_type"] == "print":
        write_print_table(full_result_rows, column_labels_full_print)
    else:
        print("Unknown table type")
    