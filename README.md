# AI-Agent

AI-Agent is a Python CLI for tabular-data analysis and machine-learning code generation. It reads CSV or ARFF files, recommends feature engineering steps, generates training code, validates the result, and can automatically retry with fixes when the first pass fails.

## What the project does now

- Reads datasets from the data folder
- Uses Azure OpenAI for structured analysis and code generation
- Recommends features to keep, scale, encode, and drop
- Produces Python machine-learning scripts in the generated_code folder
- Validates generated code with syntax and execution checks
- Automatically retries and improves code when validation finds errors
- Optionally uses local OpenML-style context for similar datasets
- Saves LLM responses for debugging in the responses folder

## Prerequisites

- Docker Desktop installed and running
- Azure OpenAI credentials
- Git

## Quick start with Docker

### 1. Clone the repository

```bash
git clone https://github.com/Lord-Sinq/AI-Agent
cd AI-Agent
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Update .env with your Azure OpenAI settings:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=your-deployment
AZURE_API_VERSION=2024-02-15-preview
DEFAULT_MODEL=gpt-4o-mini
SAVE_RESPONSES=true
```

### 3. Build and start the container

```bash
docker-compose build
docker-compose up -d
```

### 4. Run the app inside the container

```bash
docker-compose run --rm ai-agent python main.py
```

## Docker commands

```bash
docker-compose build
docker-compose up -d
docker-compose down
docker ps
docker logs ai-agent
```

## Main commands

```bash
python main.py
python main.py --file data/employees.csv
python main.py --file data/medical_patients.csv --domain healthcare
python main.py --list-deployments
python main.py --quiet --no-openml
```

## How the current workflow works

1. File selection and basic dataset inspection
2. Optional domain analysis
3. Feature engineering recommendations
4. Model code generation
5. Code validation and auto-fix retries
6. Output summary plus generated script files

## Outputs

- Console summary of selected features, target, and recommended models
- Python files written to generated_code/
- Validation details and retry history in the pipeline output
- Saved LLM responses in responses/

## Project structure

```text
AI-Agent/
├── data/                  # Example datasets
├── generated_code/        # Generated ML scripts
├── responses/             # Saved LLM responses
├── main.py                # CLI entry point
├── manager.py             # Orchestrator for the pipeline
├── feature_agent.py       # Feature engineering agent
├── modeling_agent.py      # Model generation agent
├── code_validation_agent.py
├── domain_agent.py
├── llms.py                # Azure OpenAI integration
├── requirements.txt
├── README.md
└── GUIDE.md
```

## Troubleshooting

- If Azure OpenAI is not configured, run: `python main.py --list-deployments`
- If no files are found, place CSV or ARFF files in the data folder
- If the generated code fails validation, the app will retry with feedback and save a fixed version in generated_code/
- If response logging is useful, keep SAVE_RESPONSES=true

## Requirements

- pandas>=3.0.3
- numpy>=2.4.6
- scikit-learn>=1.9.0
- xgboost>=3.2.0
- python-dotenv>=1.0.0
- requests>=2.31.0
- scipy>=1.10.0

## Notes

The repository is currently focused on a practical, end-to-end workflow: analyze a dataset, generate useful ML code, and improve that code through validation rather than stopping at a first draft.
