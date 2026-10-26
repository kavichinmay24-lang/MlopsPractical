python --version

Step 3
python -m venv venv

Activate the environment:
.\venv\Scripts\Activate.ps1

Step 4:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sklearn; print(sklearn.__version__)"
```

Expected version:
1.6.1

Step 5
python ml_pipeline.py

Step 6: 
trained_pipeline.joblib
```

