docker --version
docker info
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload
Open the interactive API documentation:
http://127.0.0.1:8000/docs
Use the POST /predict endpoint and enter this request body:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}
docker build -t ml-fastapi-app .
docker images
docker run -d -p 8000:8000 --name ml-container ml-fastapi-app
Open the API documentation: http://127.0.0.1:8000/docs
Select POST /predict, click Try it out, enter:
{
  "features": [5.1, 3.5, 1.4, 0.2]
}

