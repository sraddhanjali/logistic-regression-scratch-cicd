import yaml
import numpy as np

def load_config(config_file: str ="config.yml"):
    "Loading params from config yaml file"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

import yaml

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionaries using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def yield_yaml_params_to_mlflow(yaml_path):
    """Load YAML, flatten it, and log string params to MLflow."""
    config = load_config(config_file=yaml_path)
    flat_config = flatten_dict(config)

    for key, value in flat_config.items():
        if isinstance(value, np.number):
            yield (key, value)
            print(f"Logged: {key} = {value}")
        else:
            print(f"Skipped: {key} (non-string type: {type(value)})")


# set for global use
CONFIG = load_config()

CONFIG["random_state_"] = np.random.RandomState(CONFIG["seed"])