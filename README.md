# AI-Agent

AI-powered data organization tool that uses Azure OpenAI to intelligently sort and verify CSV and text files.

### Prerequisites

#### For Local Python (venv):

- Python 3.10 or higher
- Git
- Azure OpenAI account with API access

#### For Docker:

- Docker Desktop installed
- Azure OpenAI account with API access

### Setup Instructions

#### Option 1: Run with Docker (Recommended)

1. **Clone the repository:**

```bash
git clone "https://github.com/Lord-Sinq/AI-Agent"
cd AI-Agent
```

2. **Configure environment variables:**

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key
AZURE_DEPLOYMENT_NAME=DeepSeek-V3.2-Speciale
AZURE_API_VERSION=2024-02-15-preview
DEFAULT_MODEL=gpt-4o-mini
```

3. **Build and run:**

```powershell
docker-compose build
docker-compose run --rm ai-agent
```

### Option 2: Run with Virtual Environment (Local Development)

1. **Clone the repository:**

```bash
git clone "https://github.com/Lord-Sinq/AI-Agent"
cd AI-Agent
```

2. **Create and activate virtual environment (Windows PowerShell):**

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. **Configure environment variables:**

```bash
cp .env.example .env
```

5. **Run the application:**

```powershell
python main.py
```

## Usage

Once running, you'll see a list of available files in the `data/` directory:

```
Available files in 'data':
  1. employees.csv
  2. random.txt

Select a file (1-2):
```

Select a file by entering its number. The application will then process it through two AI agents:

- **Data Organizer Agent** - Analyzes the file structure and recommends the optimal organization strategy
- **Verifier Agent** - Reviews the organization and validates if it's the best approach

The results are displayed as JSON output showing the analysis, recommendations, and sorted preview.

## Command Line Options

```bash
python main.py --list-deployments
python main.py --file data/employees.csv
python main.py --model gpt-4o
python main.py --help
```

## Docker Commands Reference

```powershell
docker-compose build
docker-compose run --rm ai-agent
docker-compose run --rm ai-agent --list-deployments
docker-compose run --rm ai-agent --file data/employees.csv
docker-compose run --rm --entrypoint /bin/bash ai-agent
docker-compose down -v
```

## Project Structure

```
AI-Agent/
├── data/
│   ├── employees.csv
│   └── random.txt
├── main.py
├── llms.py
├── agent.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
└── README.md
```

## Adding Your Own Data Files

```bash
cp /path/to/your/file.csv data/
```

The application automatically detects all files in `data/` (excluding `.gitkeep` and hidden files).

## Environment Variables

| Variable              | Description                  | Example                                 |
| --------------------- | ---------------------------- | --------------------------------------- |
| AZURE_OPENAI_ENDPOINT | Azure OpenAI endpoint URL    | https://your-resource.openai.azure.com/ |
| AZURE_OPENAI_API_KEY  | Azure OpenAI API key         | your-api-point...                       |
| AZURE_DEPLOYMENT_NAME | Model deployment name        | DeepSeek-V3.2-Speciale                  |
| AZURE_API_VERSION     | API version to use           | 2024-02-15-preview                      |
| DEFAULT_MODEL         | Default model for generation | gpt-4o-mini                             |

# **Important:** Never commit your `.env` file to version control.

## Troubleshooting

**Docker: ".env file not found" warning**
Create `.env` from `.env.example` and add your Azure credentials.

**Environment variables not working**
Ensure your `.env` file has NO quotes around values:

```env
# Correct
AZURE_DEPLOYMENT_NAME=DeepSeek-V3.2-Speciale

# Incorrect
AZURE_DEPLOYMENT_NAME="DeepSeek-V3.2-Speciale"
```

**No files found in data directory**

- Place files in `./data/` folder
- `.gitkeep` is automatically ignored

**API connection errors**

```bash
python main.py --list-deployments
```

**Docker build fails**

```bash
docker-compose build --no-cache
```

## Clean Up

**Docker:**

```powershell
docker-compose down -v
docker rmi ai-agent-ai-agent
```

**Virtual environment:**

```powershell
deactivate
rmdir /s venv
```

## Security Notes

- Docker container runs as non-root user
- API keys never stored in Docker image
- `.env` file excluded from version control

## Requirements

- python-dotenv>=1.0.0
- requests>=2.31.0

## How It Works

1. File Detection - Scans `data/` directory for files
2. Data Organizer Agent - Sends sample to Azure OpenAI
3. AI Analysis - LLM recommends sorting strategy
4. Data Organization - Applies recommended sorting
5. Verifier Agent - Validates the organization
6. Results Display - Outputs JSON with analysis

## Acknowledgments

Built with Azure OpenAI Service. Uses DeepSeek and GPT models for intelligent data analysis.
