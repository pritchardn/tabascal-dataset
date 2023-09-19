import json
import os


def load_config_file(input_dir: str, filename: str):
    """
    Loads the config file from the input directory
    """
    config_file_path = os.path.join(input_dir, filename)
    with open(config_file_path, "r") as ifile:
        config_vals = json.load(ifile)
    return config_vals


def write_config_file(output_dir: str, config_vals: dict):
    """
    Writes the config file to the output directory
    """
    config_file_path = os.path.join(output_dir, "config.json")
    with open(config_file_path, "w") as ofile:
        json.dump(config_vals, ofile, indent=4)
