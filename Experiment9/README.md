python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dataset_tool.py
Step Run with Explicit File Paths
python dataset_tool.py --input sample_dataset.csv --output cleaned_dataset.csv
Get-Content cleaned_dataset.csv