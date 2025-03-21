import os
import glob
import subprocess
from functools import wraps
import logging
import sys
sys.path.append(".")
from utils.mlflow_manager import MLflowManager

LOG_FILE = "dvc_manager.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)

logger = logging.getLogger("dvc-manager")

def ensure_dvc_initialized(func):
    """Decorator to initialize DVC repo if not already."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not os.path.exists(".dvc"):
            print("🛠 Initializing DVC...")
            subprocess.run(["dvc", "init"], check=True)
        else:
            print("✅ DVC already initialized.")
        return func(*args, **kwargs)
    return wrapper

def skip_if_tracked(func):
    """Skip DVC add if file is already tracked."""
    @wraps(func)
    def wrapper(d, *args, **kwargs):
        if os.path.exists(d + ".dvc"):
            print(f"✅ File '{d}' already tracked by DVC. Skipping.")
            return
        return func(d, *args, **kwargs)
    return wrapper

def ensure_remote_not_exists(func):
    """Decorator to skip remote creation if it already exists."""
    @wraps(func)
    def wrapper(storage, name, *args, **kwargs):
        result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
        print(f"Type of the file is {type(name)}")
        if name in result.stdout:
            print(f"✅ Remote '{name}' already exists. Skipping.")
            return
        return func(storage, name, *args, **kwargs)
    return wrapper


def get_all_nested_files(func):
    """Decorator to apply `func` to each .csv file in directory, or just once if single file."""
    @wraps(func)
    def wrapper(d, *args, **kwargs):
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, "**/*.csv"), recursive=True)
            for f in files:
                func(f, *args, **kwargs)
        elif os.path.isfile(d):
            func(d, *args, **kwargs)
    return wrapper

class DVCManager:

    dvc_config = None

    def run_command(cmd):
        logger.info(f"💻 Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command failed: {' '.join(cmd)}")
            pass

    @staticmethod
    @ensure_dvc_initialized
    @ensure_remote_not_exists
    def add_remote(storage, name):
        name = name
        url = storage
        cmd = ["dvc", "remote", "add", "-d", name, url]
        DVCManager.run_command(cmd)
        print(f"🚀 DVC remote '{name}' added.")
    
    @staticmethod
    def setup_remote(dvc_config=None): 
        dvc_config=dvc_config
        if f"{dvc_config['remote']}":
            DVCManager.add_remote(dvc_config["remote_storage"], "myremote")
            print(f"🚀 myremote DVC remote added.")
        else:
            DVCManager.add_remote(dvc_config["local_storage"], "localremote")
            print(f"🚀 local_remote DVC remote added.")
    
    @staticmethod
    @get_all_nested_files
    @skip_if_tracked
    def version(d):
        dvc_add = ["dvc", "add", f"{d}"]
        git_add = ["git", "add", f"{d}.dvc"]
        DVCManager.run_command(dvc_add)
        DVCManager.run_command(git_add)
       
    @staticmethod
    def commit_and_push():
        git_commit = ["git", "commit", "-m", "'Version dataset'"]
        dvc_push = ["dvc", "push"]
        DVCManager.run_command(git_commit)
        DVCManager.run_command(dvc_push)

    @staticmethod
    def log_to_mlflow(mlflow_obj: MLflowManager):
        if os.path.exists(LOG_FILE):
            mlflow_obj.log({"artifacts": LOG_FILE})