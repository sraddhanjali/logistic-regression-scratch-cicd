import subprocess
import mlflow
import sys
import os
sys.path.append(".")
from utils.config import load_config
import time

class MLflowManager:
    def __init__(self, mlflow_config=None):
        self.mlflow_config = mlflow_config
        self.log_file = os.path.join(os.getcwd(), self.mlflow_config["log_file"])
        self.process = None  # Store the MLflow process
        self.mlflow_uri = self.mlflow_config["tracking_uri"]
        self.mlflow_experimentname = self.mlflow_config["experiment_name"]
        self.mlflow_port = self.mlflow_config["default_port"]

    def start(self):
        """Start MLflow server in the background and keep track of it."""
        mlflow.set_tracking_uri(f"http://localhost:{self.mlflow_port}")
        exp = mlflow.set_experiment(self.mlflow_experimentname)       
        print(f'experiment starting {exp.experiment_id}')
        log = open(self.log_file, "a")
        
        # Start MLflow as a background process
        self.process = subprocess.Popen([
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", str(self.mlflow_port)
        ], stdout=log, stderr=log, start_new_session=True)

        time.sleep(5)  # Give MLflow time to start
        print(f"✅ MLflow server started on port {self.mlflow_port}. Logs: {self.log_file}")
    
    def log(self, info=None):
        if not info:
            mlflow.log_param("no info", 1)
            return
        for k, v in info.get("params", {}).items():
            mlflow.log_param(k, v)

        for k, v in info.get("metrics", {}).items():
            mlflow.log_metric(k, v)

        for path in info.get("artifacts", []):
            mlflow.log_artifact(path)

        # Optional config logging
        if "configs" in info:
            for k, v in info["configs"].items():
                mlflow.log_param(f"{k}", v)
    
    def stop(self):
        """Stop the MLflow server if it's running."""
        if self.process:
            print("🛑 Stopping MLflow server...")
            self.process.terminate()
            self.process.wait()
            print("✅ MLflow server stopped.")
        else:
            print("⚠️ No MLflow process to stop.")

if __name__ == "__main__":
    mlflow_server = MLflowManager(mlflow_config=load_config())
    mlflow_server.start()
    mlflow_server.log()
    mlflow_server.stop()