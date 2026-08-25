# AI-Agent: Intelligent Data Science Pipeline

## Executive summary

AI-Agent is a Python-based workflow for analyzing tabular data, recommending feature engineering steps, generating machine-learning code, and validating that code with automatic retry logic. The project is designed to show how an AI-assisted pipeline can move from raw data to a runnable Python model script.

## What the project does today

- Reads CSV and ARFF files from the data directory
- Uses Azure OpenAI to produce structured recommendations
- Recommends features to keep, scale, encode, and drop
- Generates Python model-training scripts in the generated_code folder
- Validates generated code for syntax and execution problems
- Retries automatically with feedback if the first version fails
- Optionally uses local OpenML-style context for similar datasets

## Current architecture

```text
User input
  ↓
main.py
  ↓
Manager / orchestration layer
  ├─ Feature engineering agent
  ├─ Modeling agent
  ├─ Domain analysis agent
  ├─ Code validation agent
  └─ OpenML context lookup
  ↓
Console output + generated code + response logs
```

## Core components

- main.py: CLI entry point and argument handling
- manager.py: Coordinates the full workflow
- feature_agent.py: Produces feature recommendations
- modeling_agent.py: Generates machine-learning code
- code_validation_agent.py: Checks generated scripts for errors
- llms.py: Handles Azure OpenAI requests and response storage

## Current workflow

1. Select a dataset from data/
2. Optionally provide a domain or target context
3. Generate feature engineering suggestions
4. Create model-training code
5. Validate the code and retry if needed
6. Save the best result to generated_code/

## Key talking points

- This project turns a dataset into a usable ML workflow quickly.
- The pipeline is practical and transparent rather than purely theoretical.
- The generated scripts are meant to be editable and runnable.
- The repository demonstrates a complete AI-assisted development loop: generate, validate, and improve.

## Example command

```bash
python main.py --file data/medical_patients.csv --domain healthcare
```

## Project outputs

- Console summary of the run
- Generated Python files in generated_code/
- Saved responses in responses/
- Validation results and retry history in the pipeline output
