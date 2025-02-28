import yaml
import numpy as np

def load_config(config_file: str ="config.yml"):
    "Loading params from config yaml file"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

# set for global use
CONFIG = load_config()

CONFIG["random_state_"] = np.random.RandomState(CONFIG["seed"])