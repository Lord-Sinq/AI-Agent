# AI-Agent: Quick Guide

## Run the project with Docker

### 1. Prepare the environment

```bash
git clone https://github.com/Lord-Sinq/AI-Agent
cd AI-Agent
cp .env.example .env
```

Fill in your Azure OpenAI values before running the container.

### 2. Build and start the container

```bash
docker-compose build
docker-compose up -d
```

### 3. Start the CLI inside the container

```bash
docker-compose run --rm ai-agent python main.py
```

The app will show the available datasets in the data folder and ask you to choose one.

## Example commands

```bash
docker-compose run --rm ai-agent python main.py --file data/medical_patients.csv --domain healthcare
docker-compose run --rm ai-agent python main.py --file data/employees.csv --no-openml
docker-compose run --rm ai-agent python main.py --list-deployments
docker-compose run --rm ai-agent python main.py --save-responses --quiet
```

## What to expect

When the run completes, you should see:

- a summary of the selected features and target
- recommended model types
- a generated Python script in generated_code/
- any validation issues and auto-fix attempts
- saved responses in responses/ if enabled

## Troubleshooting

### Azure OpenAI connection error

```bash
python main.py --list-deployments
```

Check that your .env file contains the correct Azure endpoint, key, deployment name, and API version.

### No files found in data

Make sure your CSV or ARFF files are placed under the data folder before launching the application.

### Validation or generation failed

The pipeline is designed to retry automatically. Review the generated files in generated_code/ and the saved response logs in responses/.
