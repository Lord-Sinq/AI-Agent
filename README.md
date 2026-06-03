# AI-Agent

AI-powered data organization tool that uses Azure OpenAI to intelligently sort and verify CSV and text files.

## Prerequisites

- Docker Desktop installed and running (recommended)
- Azure OpenAI account with API access
- Git

## Quick Start with Docker

### 1. Clone the repository

git clone https://github.com/Lord-Sinq/AI-Agent
cd AI-Agent

### 2. Set up environment variables

cp .env.example .env

Edit .env with your Azure OpenAI credentials:

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key
AZURE_DEPLOYMENT_NAME=DeepSeek-V3.2-Speciale
AZURE_API_VERSION=2024-02-15-preview
DEFAULT_MODEL=gpt-4o-mini

### 3. Build and start the container

docker-compose build
docker-compose up -d

### 4. Attach VS Code to the container

- Open VS Code
- Click the remote button (><) in the bottom-left corner
- Select "Attach to Running Container..."
- Choose ai-agent from the list
- Open the /app folder
- Open a terminal with Ctrl+`

### 5. Run the application

python main.py

### 6. Stop the container

docker-compose down

## Run Without VS Code

docker-compose run --rm ai-agent

## Usage

Once running, select a file from the data/ directory:

Available files in 'data':

1. employees.csv
2. random.txt

Select a file (1-2):

The application processes your file through two AI agents:

- Data Organizer Agent - Analyzes and recommends organization strategy
- Verifier Agent - Validates the approach

## Command Line Options

python main.py --list-deployments
python main.py --file data/employees.csv
python main.py --model gpt-4o
python main.py --help

## Docker Commands

docker-compose build # Build the container
docker-compose up -d # Start container in background
docker-compose down # Stop container
docker-compose run --rm ai-agent # Run once and exit
docker ps # List running containers
docker logs ai-agent # View container logs

### Local Python Development (Without Docker)

#### 1. Clone and create virtual environment

git clone https://github.com/Lord-Sinq/AI-Agent
cd AI-Agent
python -m venv venv

#### 2. Activate virtual environment

Windows PowerShell:
.\venv\Scripts\Activate.ps1

Windows Command Prompt:
venv\Scripts\activate.bat

Mac/Linux:
source venv/bin/activate

#### 3. Install dependencies

pip install -r requirements.txt

#### 4. Set up environment variables

```
cp .env.example .env
```

##### Edit .env with your credentials

#### 5. Open in VS Code

VS Code will prompt you to install recommended extensions (Python, Pylance, etc.)

#### 6. Run or debug

- Press F5 to debug main.py
- Or run: python main.py

## Project Structure

AI-Agent/ \
├── data/ \
│ ├── employees.csv
│ └── random.txt \
├── main.py \
├── llms.py \
├── agent.py \
├── requirements.txt \
├── Dockerfile \
├── docker-compose.yml \
├── .env.example \
└── README.md \

## Adding Your Own Data Files

cp /path/to/your/file.csv data/

## Environment Variables

Variable: AZURE_OPENAI_ENDPOINT
Description: Azure OpenAI endpoint URL
Example: https://your-resource.openai.azure.com/

Variable: AZURE_OPENAI_API_KEY
Description: Azure OpenAI API key
Example: your-api-key

Variable: AZURE_DEPLOYMENT_NAME
Description: Model deployment name
Example: DeepSeek-V3.2-Speciale

Variable: AZURE_API_VERSION
Description: API version
Example: 2024-02-15-preview

Variable: DEFAULT_MODEL
Description: Default model
Example: gpt-4o-mini

IMPORTANT: Never commit your .env file to version control.

## Troubleshooting

Docker: ".env file not found"

- Create .env from .env.example

Environment variables not working

- Ensure no quotes around values in .env

No files found in data directory

- Place files in ./data/ folder

API connection errors
python main.py --list-deployments

Docker build fails
docker-compose build --no-cache

## Clean Up

docker-compose down -v
docker rmi ai-agent-ai-agent

## Security

- Container runs as non-root user
- API keys never stored in Docker image
- .env excluded from version control

## Requirements

- python-dotenv>=1.0.0
- requests>=2.31.0

## How It Works

1. File Detection - Scans data/ directory for files
2. Data Organizer Agent - Sends sample to Azure OpenAI
3. AI Analysis - LLM recommends sorting strategy
4. Data Organization - Applies recommended sorting
5. Verifier Agent - Validates the organization
6. Results Display - Outputs JSON with analysis

## Acknowledgments

Built with Azure OpenAI Service. Uses DeepSeek and GPT models for intelligent data analysis.
