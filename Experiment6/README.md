python --version
python -m venv venv
.\venv\Scripts\Activate.ps1

If activation is blocked, use .\venv\Scripts\python.exe for installation and execution commands.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Step 4
python train_model.py

Step 5
python app.py

The application starts at:

http://127.0.0.1:5000
Keep this terminal open while testing the API.

Step 6: Test the Home Endpoint
Open this URL in your browser:

http://127.0.0.1:5000

Open a second PowerShell terminal in the same project folder.
Execute:

$body = @{
    features = @(5.1, 3.5, 1.4, 0.2)
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body