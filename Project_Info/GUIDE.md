# AI-Agent: Quick Guide

## Setup

Use Python 3.13 or the included Dockerfile. Install dependencies and configure Azure OpenAI:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_DEPLOYMENT_NAME` in `.env`. `STRICT_JSON_MODE=true` is the recommended default. Response saving can be enabled with `SAVE_RESPONSES=true`.

## Run the CLI

Start interactively to choose a dataset from `data/`:

```bash
python main.py
```

Or provide a current repository dataset directly:

```bash
python main.py --file data/dataset_titanic.arff
python main.py --file data/dataset_titanic.arff --target class --problem classification
python main.py --file data/dataset_194_eucalyptus.arff --domain ecology --no-openml
```

### CLI controls

- `-f`, `--file`: use a specific CSV or ARFF dataset instead of choosing one interactively.
- `-m`, `--model`: select the Azure OpenAI deployment or model to use.
- `-d`, `--domain`: provide a domain such as healthcare, retail, or ecology for additional context.
- `-t`, `--task`: describe the analysis or machine-learning task.
- `--target`: specify the target column for prediction.
- `--problem`: set the problem type to `classification`, `regression`, or `clustering`.
- `--list-deployments`: list the Azure deployments available to the configured account.
- `--save-responses`: save LLM responses for debugging under `responses/`.
- `--no-save-responses`: disable LLM response saving for the current run.
- `-q`, `--quiet`: reduce console output and skip the response-saving prompt.
- `--no-openml`: skip the local similarity search in `openMLdatasets/`.

Use `python main.py --help` for the complete list. The CLI currently asks for an optional data description after selecting a file; press Enter to leave it blank.

## Run the dashboard (working progress/not done)

The Textual terminal dashboard provides dataset selection, configuration controls, full pipeline execution, individual debug-step buttons, and a results view:

```bash
python ml_dashboard.py
```

Select a dataset, configure the optional fields, and choose **RUN FULL PIPELINE**. The dashboard can also list Azure deployments and toggle local OpenML context or response saving.

## Run with Docker

```bash
docker-compose build
docker-compose up -d
docker exec -it ai-agent bash
python main.py
```

Compose keeps the development container running and mounts the repository. The active mount is the macOS entry in `docker-compose.yml`; enable the Linux `.:/app` entry when running on Linux. The mounted source is required because the Dockerfile does not currently copy every pipeline module into the image.

Stop the container with:

```bash
docker-compose down
```

## Pipeline results

The pipeline performs domain analysis when requested, checks the local `openMLdatasets/` index for similar data, designs features, generates model code, validates syntax and execution, retries failed code generation with feedback, and runs performance validation.

Expect to see:

- selected feature count, problem type, target, and recommended models
- validation status, issues, score, retry attempts, and performance improvement
- generated code path and a short code preview in the CLI
- total runtime

Generated scripts are written to `generated_code/`. A corrected retry may be saved as `<dataset>_model_fixed.py`. When response saving is enabled, LLM and code-validator responses are written to `responses/<date>/`.

## Troubleshooting

### Azure configuration

```bash
python main.py --list-deployments
```

Verify the endpoint, API key, deployment name, and API version in `.env`.

### No datasets found

Put a `.csv` or `.arff` file in `data/`. The current repository examples are ARFF files, including `dataset_titanic.arff` and `dataset_11_balance-scale.arff`.

### ARFF loading fails

Ensure the dependencies installed successfully, especially `scipy`, which is used for ARFF parsing.

### Generated code fails validation

Review the generated scripts and saved responses. The manager automatically retries code generation, up to five attempts by default, using the validator's error feedback.
