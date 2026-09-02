# AI-Agent

AI-Agent is a Python application for tabular-data analysis and machine-learning code generation. It supports an interactive CLI and working on a Textual terminal dashboard (TUI). The pipeline reads CSV or ARFF files, uses Azure OpenAI for structured analysis, generates training code, validates it, and retries failed generations with error feedback.

## Features

- Dataset selection and inspection from `data/`
- Optional domain-specific analysis
- Feature recommendations for keeping, scaling, encoding, dropping, and deriving columns
- Local OpenML-style similarity context from `openMLdatasets/`
- Generated model scripts in `generated_code/`
- Syntax, execution, and model-performance validation
- Automatic code-fix retries, with up to five attempts by default
- Optional structured response logging in `responses/<date>/`
- Strict JSON output mode for LLM responses

## Requirements

- Python 3.13 recommended (the Dockerfile uses `python:3.13-slim`)
- Azure OpenAI endpoint, API key, and deployment
- Git
- Docker Desktop, if using the container workflow

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The current dependencies include `pandas`, `numpy`, `scipy`, `scikit-learn`, `requests`, `python-dotenv`, and `textual`.

## Configuration

Copy the example environment file and fill in the Azure values:

```bash
cp .env.example .env
```

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=your-deployment
AZURE_API_VERSION=2024-02-15-preview
DEFAULT_MODEL=gpt-4o-mini
STRICT_JSON_MODE=true
```

`SAVE_RESPONSES` can be set to `true` or `false`. It can also be controlled with `--save-responses` and `--no-save-responses`; in interactive CLI mode, the application asks when no value is configured. `OPENML_DATASETS_DIR` can override the default local dataset directory `openMLdatasets/`.

The OpenML agent currently uses the local dataset index and does not require a network lookup. The optional `OPENML_API_KEY`, `OPENML_TIMEOUT`, and `OPENML_MAX_RETRIES` values in `.env.example` are retained for related OpenML tooling.

## Run locally

The CLI scans `data/` for `.csv` and `.arff` files. The repository currently includes example ARFF datasets such as `dataset_11_balance-scale.arff` and `dataset_titanic.arff`.

```bash
python main.py
python main.py --file data/dataset_titanic.arff
python main.py --file data/dataset_titanic.arff --domain healthcare --target class
python main.py --file data/dataset_titanic.arff --problem classification --model gpt-4o-mini
python main.py --list-deployments
python main.py --file data/dataset_titanic.arff --no-openml --no-save-responses
python main.py --file data/dataset_titanic.arff --hyperparameters config/hyperparameters.json
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
- `--hyperparameters`: use a JSON file mapping model names to constructor parameters. Defaults to `config/hyperparameters.json`.

The data-description prompt is currently shown after file selection even when `--quiet` is supplied, so provide input when running the CLI interactively.

## Textual dashboard

Launch the terminal UI with:

```bash
python ml_dashboard.py
```

The dashboard lets you select a dataset, set the target/domain/model, choose a problem type, enable or disable local OpenML context and response saving, run the full pipeline, run debug steps, and inspect results. It uses `style.tcss` for its layout and styling.

## Docker development workflow

```bash
docker-compose build
docker-compose up -d
docker exec -it ai-agent bash
python main.py
```

The Compose service keeps the container alive with `tail -f /dev/null` and mounts the repository for development. The active volume entry is configured for macOS; on Linux, enable the Linux `.:/app` volume in `docker-compose.yml` before running the container. The image currently copies only a subset of the application modules, so the mounted repository is required for the full pipeline.

Stop the service with:

```bash
docker-compose down
```

## Workflow and outputs

1. Select or provide a dataset path and optional user description.
2. Optionally analyze the domain and find similar local OpenML datasets.
3. Design feature engineering steps.
4. Generate model code.
5. Validate syntax and execution, then retry with feedback when needed.
6. Run the performance validation stage and print a summary.

Generated scripts are saved in `generated_code/`. A retry may create both `<dataset>_model.py` and `<dataset>_model_fixed.py`. The CLI also prints the generated code preview, validation summary, response directory when enabled, and total runtime. Structured LLM and validator responses are saved under `responses/` when response saving is enabled.

### Hyperparameter configuration

Model settings are stored in `config/hyperparameters.json` and included in the modeling and code-fix prompts. Each key is a model class name and each value is a JSON object of constructor parameters. Use `--hyperparameters PATH` to run an experiment with a different configuration. The selected configuration is included in the pipeline result for reproducibility.

## Project structure

```text
AI-Agent/
├── data/                  # Input CSV and ARFF datasets
├── openMLdatasets/        # Local OpenML-style datasets and index
├── generated_code/        # Generated ML scripts
├── responses/             # Saved LLM responses by date
├── main.py                # CLI entry point
├── ml_dashboard.py        # Textual terminal dashboard
├── manager.py             # Pipeline orchestrator
├── domain_agent.py        # Domain analysis
├── feature_agent.py       # Feature engineering
├── modeling_agent.py      # Model code generation
├── code_validation_agent.py
├── openMLAgentLocal.py    # Local similarity lookup
├── llms.py                # Azure OpenAI integration
├── style.tcss             # Dashboard styling
└── requirements.txt
```

## Troubleshooting

- Run `python main.py --list-deployments` to test Azure configuration and list deployments.
- If no datasets are found, place a `.csv` or `.arff` file in `data/`.
- If ARFF files cannot be read, reinstall the dependencies so `scipy` is available.
- If generated code fails validation, inspect the retry history and the generated files; the pipeline automatically sends validation errors back to the model.
- Set `SAVE_RESPONSES=true` or pass `--save-responses` when debugging model and validator output.
