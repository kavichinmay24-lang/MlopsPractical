python --version
python -m venv venv

.\venv\Scripts\Activate.ps1
```

If activation is blocked, use `.\venv\Scripts\python.exe` in place of `python` for the following commands.

Step 3

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mlflow --version


Step 4

python mlflow_experiment.py

Step 5

mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

Open this address in your browser:

http://127.0.0.1:5000

Step 6

python mlflow_experiment.py

Refresh the MLflow UI


