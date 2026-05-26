# AI-Agent

Minimal instructions to set up and run the project locally.

Prerequisites

- Python 3.10+ and Git

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
python main.py
```
From the main file you can pick from a list of data files stored in the data folder.
After picking a file it will be passed though both verifier and data constructor agent. 
