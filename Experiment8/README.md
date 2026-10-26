python --version
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python onnx_conversion.py
Test-Path model.onnx