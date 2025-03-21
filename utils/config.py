import yaml
import numpy as np

def load_config(config_file: str ="config.yml"):
    "Loading params from config yaml file"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def flatten_numeric_values(d, parent_key='', sep='.'):
    """Flatten nested dictionaries using dot notation."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_numeric_values(v, new_key, sep=sep))
        elif isinstance(v, (int, float)):
            items[new_key] = v
    return items

# set for global use
CONFIG = load_config()

CONFIG["random_state_"] = np.random.RandomState(CONFIG["seed"])