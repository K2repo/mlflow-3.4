# experiments/mlflow_config.py
import mlflow
import os

# Configurar el tracking URI (puede ser local o remoto)
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Para empezar, usa SQLite

# O para un servidor remoto (recomendado para producción):
# mlflow.set_tracking_uri("http://localhost:5000")