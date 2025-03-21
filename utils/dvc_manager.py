import os
import glob
import subprocess
from functools import wraps

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
    def wrapper(filepath, *args, **kwargs):
        if os.path.exists(filepath + ".dvc"):
            print(f"✅ File '{filepath}' already tracked by DVC. Skipping.")
            return
        return func(filepath, *args, **kwargs)
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
    """Decorator to yield files if dir is provided and not being tracked."""
    @wraps(func)
    def wrapper(_path, *args, **kwargs):
        if os.path.isdir(_path):
            files = glob.glob(f"{_path}/*.csv", recursive=True)
            for f in files:
                yield f
            yield func(f, *args, **kwargs)
        if os.path.isfile(_path):
            return func(_path, *args, **kwargs)
    return wrapper

class DVCManager:
    def __init__(self, dvc_config=None):
        self.dvc_config = dvc_config

    @staticmethod
    @skip_if_tracked
    def version_data(f_path):
        subprocess.run(["dvc", "add", f_path], check=True)
        subprocess.run(["git", "add", f"{f_path}.dvc", ".gitignore"])
        os.system("git commit -m 'Version dataset'")
        os.system("dvc push")

    @staticmethod
    @ensure_dvc_initialized
    @ensure_remote_not_exists
    def add_remote(storage, name):
        name = name
        url = storage
        subprocess.run(["dvc", "remote", "add", "-d", name, url], check=True)
        print(f"🚀 DVC remote '{name}' added.")
    
    def setup_remote(self):
        if f"{self.dvc_config['remote']}":
            DVCManager.add_remote(self.dvc_config["remote_storage"], "myremote")
            print(f"🚀 myremote DVC remote added.")
        else:
            DVCManager.add_remote(self.dvc_config["local_storage"], "localremote")
            print(f"🚀 local_remote DVC remote added.")

    
    @get_all_nested_files
    @skip_if_tracked
    def version(self, d):
        subprocess.run(["dvc", "add", f"{d}"])
        subprocess.run(["git", "add", f"{d}"])
    
    def push_all(self):
        subprocess.run(["dvc", "push"])
