Open a terminal inside Experiment7_CI_ML_Project:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v
python train_model.py
git add .github/workflows/exp7-ci.yml Experiment7_CI_ML_Project
git status
git commit -m "Add Experiment 7 GitHub Actions CI"
git push
use git push -u origin main instead of git push
git config --global user.name "Your GitHub Name"
git config --global user.email "your-github-email@example.com"