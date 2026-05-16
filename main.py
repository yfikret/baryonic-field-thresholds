import os
import yaml

from src.visualization import make_plots
from src.metrics import run_stats

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    yaml_path = os.path.join(BASE_DIR, "config", "config.yaml")
    with open(yaml_path, 'r') as file:
        CONFIG = yaml.safe_load(file)
    
    CONFIG["save_plots_path"] = os.path.join(BASE_DIR, CONFIG["save_plots_path"])
    CONFIG["save_markdown_path"] = os.path.join(BASE_DIR, CONFIG["save_markdown_path"])
    CONFIG["SPARC_archive_path"] = os.path.join(BASE_DIR, CONFIG["SPARC_archive_path"])
    CONFIG["SPARC_parsed_path"] = os.path.join(BASE_DIR, CONFIG["SPARC_parsed_path"])
    CONFIG["profiles_path"] = os.path.join(BASE_DIR, CONFIG["profiles_path"])
    CONFIG["latex_output_path"] = os.path.join(BASE_DIR, CONFIG["latex_output_path"])
    
    print("")
    
    if CONFIG["mode"] == "make_plots":
        print(f"Running plotter for window {CONFIG["local_window"]}")
        
        make_plots(CONFIG)

    elif CONFIG["mode"] == "run_stats":
        print("Running stats calculation")

        run_stats(CONFIG)

if __name__ == "__main__":
    main()