
Step 2:

python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked, use `.\venv\Scripts\python.exe` instead of `python` in the following installation commands.

Step 3

python -m pip install --upgrade pip
python -m pip install dvc
dvc --version
```

Step 4
git init

git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

Step 5
dvc init --subdir
```

Step 6 

git rm --cached data.csv
```

Step 7 
cd
dvc add data.csv
```


Step 8:

git add .dvc data.csv.dvc .gitignore
git commit -m "Track first dataset version using DVC"
```

Step 9: 
(Get-Content data.csv) -replace '101,Amit,CSE,85','101,Amit,CSE,88' | Set-Content data.csv
```

Step 10: 

dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "Update dataset version"
```

### Step 11: View the Commit History

git log --oneline\

Step 12

git log --oneline
Get-Content data.csv




