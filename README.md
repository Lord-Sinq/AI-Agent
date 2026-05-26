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

Run the application:

```powershell
c:/Users/xxsta/Documents/GitHub/AI-Agent/venv/Scripts/python.exe .\main.py
```

The API will be available at http://0.0.0.0:8000 by default.
