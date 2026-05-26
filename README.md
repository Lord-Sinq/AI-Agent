# AI-Agent

Minimal instructions to set up and run the project locally.

Prerequisites

- Python 3.8+ and Git

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.
& .\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application from the venv:

```powershell
# Data agent: organize a data file using the LLM
c:/Users/xxsta/Documents/GitHub/AI-Agent/venv/Scripts/python.exe .\main.py --agent data --file data/employees.csv --task organize --provider echo

# Verifier agent: review whether the data is organized in the best way
c:/Users/xxsta/Documents/GitHub/AI-Agent/venv/Scripts/python.exe .\main.py --agent verifier --file data/employees.csv --task verify --provider echo
```
