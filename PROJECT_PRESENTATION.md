# AI-Agent: Intelligent Data Science Pipeline

## Executive Summary

**AI-Agent** is an AI-powered data analysis and machine learning code generation tool that leverages Azure OpenAI to intelligently analyze CSV files and automatically generate production-ready Python code for data science workflows. It streamlines the entire data science pipeline from exploration to model generation.

---

## Project Overview

### What It Does

AI-Agent automates the tedious parts of data science by:

- **Analyzing** CSV datasets intelligently
- **Engineering** features with ML-ready recommendations
- **Generating** complete Python ML training code
- **Validating** data quality and feature engineering decisions
- **Providing** domain-specific insights and context

### Key Value Proposition

Instead of manually exploring data, cleaning datasets, and writing boilerplate ML code, users can:

1. Upload a CSV file
2. Let AI agents analyze and generate code
3. Get production-ready Python scripts in seconds

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Main Entry Point                     │
│                   (main.py / CLI)                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐   ┌─────────▼────────────┐
│  File Picker     │   │  LLM Manager         │
│  & Validation    │   │  (Azure OpenAI)      │
└───────┬──────────┘   └──────────────────────┘
        │
        │ CSV File
        │
┌───────▼──────────────────────────────────────────┐
│           Agent Manager (Orchestrator)           │
└───────┬──────────────────────────────────────────┘
        │
        ├────────────────┬─────────────────┬───────────┐
        │                │                 │           │
   ┌────▼────┐    ┌──────▼─────┐    ┌──────▼────┐  ┌───▼─────────┐
   │ Feature │    │   Modeling │    │  Domain   │  │  OpenML     │
   │Engineer │    │   Agent    │    │  Expert   │  │  Agent      │
   │ Agent   │    │            │    │  Agent    |  |(On to start)|
   |         |    |            |    |(Optional) │  │ (Optional)  │
   └────┬────┘    └──────┬─────┘    └──────┬────┘  └────┬────────┘
        │                │                 │            │
        │Recommendations │ Model Code      │ Insights   │ Context
        │                │                 │            │
        └───────────────┬┴────────────┬────┴────────────┘
                        │             │
                  ┌─────▼─────────────▼──────┐
                  │   Result Aggregation     │
                  │   & Output Generation    │
                  └──────┬───────────────────┘
                         │
              ┌──────────┴─────────┬────────────────┐
              │                    │                | Can be up to 4 json files
         ┌────▼──────┐        ┌────▼─────┐     ┌────▼─────┐
         │  Console  │        │ Generated│     | Generated|
         │  Output   │        │  Code    │     | agent    |
         └───────────┘        │ (Python) │     | response |
                              └──────────┘     | (json)   |
                                               └──────────┘
```

---

## Workflow Pipeline

### Step 1: **File Selection & Validation**

- User selects a CSV file from the `data/` directory
- System reads file and extracts:
  - Column headers
  - Data types
  - Row count
  - Sample data preview

### Step 2: **Feature Engineering Analysis**

**Agent**: `FeatureEngineerAgent`

The AI analyzes the dataset and recommends:

- **Features to keep** (relevant columns for ML)
- **Features to scale** (normalization of numeric columns)
- **Features to encode** (categorical variables → numeric)
- **Features to drop** (IDs, metadata, irrelevant columns)
- **Derived features** (new calculated features)
- **Feature engineering code** (pandas transformations)

**Output Structure Example:**

```json
{
  "features": ["age", "salary", "experience"],
  "scale": ["age", "salary"],
  "encode": { "gender": "label", "department": "onehot" },
  "drop": ["customer_id", "timestamp"],
  "feature_code": "df['experience_per_age'] = df['experience'] / df['age']",
  "feature_metadata": { "reason": "high correlation with target" }
}
```

### Step 3: **Machine Learning Model Generation**

**Agent**: `ModelingAgent`

The AI generates:

- **Model recommendations** (classification, regression, clustering)
- **Complete Python training code**
- **Hyperparameter configurations**
- **Evaluation metrics setup**
- **Model serialization** (save trained models)

**Output:**

- Python file saved to `generated_code/` directory
- Complete, runnable ML pipeline
- Can train models immediately

### Step 4: **Domain-Specific Analysis (Optional)**

**Agent**: `DomainExpertAgent`

If a domain is specified (e.g., "healthcare", "retail"):

- Domain-specific insights
- Key metrics for the industry
- Data quality issues to watch
- Relevant business questions
- Domain-specific context

### Step 5: **OpenML Context Lookup (Optional but active unless stated)**

**Agent**: `OpenMLAgent`

Automatically searches OpenML for similar datasets:

- Similar datasets already analyzed
- Community insights
- Benchmark comparisons
- Best practices from similar problems

---

## Core Components

### 1. **LLMManager** (`llms.py`)

- Handles all communication with Azure OpenAI API
- Manages API credentials and endpoints
- Enforces strict JSON output mode
- Saves LLM responses for debugging
- Supports multiple models (DeepSeek-V3.2, GPT-4o, etc.)

### 2. **Agent Base Class** (`agent.py`)

- **File I/O**: Reads and parses CSV files
- **MIME Detection**: Identifies file types
- **Structure Extraction**: Analyzes data structure
- **JSON Parsing**: Extracts structured data from LLM responses
- **Error Handling**: Robust retry logic for failed LLM calls

### 3. **Feature Engineer Agent**

- Analyzes column data types
- Recommends scaling and encoding strategies
- Generates feature engineering code
- Validates column names against actual data

### 4. **Modeling Agent**

- Generates complete scikit-learn/XGBoost code
- Creates reproducible ML pipelines
- Includes data splitting and cross-validation
- Outputs ready-to-run Python files

### 5. **Manager (Orchestrator)** (`agent.py`)

- Coordinates all agents
- Manages workflow execution
- Aggregates results
- Formats output for users

### 6. **Feature Validator** (`caafeValidator.py`)

- CAAFE validation framework
- Ensures feature engineering quality
- Validates recommendations against data

---

## Usage Examples

### Interactive Mode

```bash
python main.py
```

- System displays available CSV files
- User selects file interactively
- AI processes and generates code

### Command-Line Mode

```bash
# Specify file and domain
python main.py --file data/sales.csv --domain retail

# Use specific model
python main.py --file data/medical_patients.csv --model gpt-4o

# Disable OpenML lookup
python main.py --file data/employees.csv --no-openml

# Save responses for debugging
python main.py --save-responses --quiet
```

### Available Flags

| Flag               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `--file PATH`      | Path to CSV file                                     |
| `--model NAME`     | LLM model to use                                     |
| `--domain DOMAIN`  | Domain context (healthcare, retail, etc.)            |
| `--target COLUMN`  | Target variable for prediction                       |
| `--problem TYPE`   | Problem type: classification, regression, clustering |
| `--no-openml`      | Disable OpenML dataset lookup                        |
| `--save-responses` | Save all LLM responses                               |
| `--quiet`          | Suppress interactive prompts                         |

---

## Data Inputs

Supported CSV files in `data/` directory:

| File                     | Rows | Columns | Purpose                       |
| ------------------------ | ---- | ------- | ----------------------------- |
| `employees.csv`          | -    | -       | HR analytics                  |
| `customer_churn.csv`     | -    | -       | Customer retention prediction |
| `medical_patients.csv`   | 101  | 9       | Healthcare analysis           |
| `sales_transactions.csv` | 21   | 9       | Sales forecasting             |
| `real_estate.csv`        | 21   | 9       | Property valuation            |
| `weather_data.csv`       | 21   | 8       | Climate prediction            |

---

## Outputs

### 1. Console Output

```
=== FEATURE ENGINEERING RECOMMENDATIONS ===
Features: [age, salary, experience]
Scale: [age, salary]
Encode: {gender: label, dept: onehot}
Drop: [customer_id]

=== MODEL RECOMMENDATIONS ===
Problem Type: Classification
Suggested Models: XGBoost, RandomForest
Code Generated: generated_code/sales_model.py

=== DOMAIN INSIGHTS (If applicable) ===
Domain: Retail
Key Metrics: [conversion_rate, avg_order_value, customer_lifetime_value]
```

### 2. Generated Python Code

File: `generated_code/{dataset_name}_model.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load data
df = pd.read_csv('data/customers.csv')

# Feature engineering
df['age_squared'] = df['age'] ** 2

# Scale and encode
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

encoder = LabelEncoder()
df['gender_encoded'] = encoder.fit_transform(df['gender'])

# Train model
X = df[['age', 'salary', 'gender_encoded', 'age_squared']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")
```

### 3. Saved Responses

Directory: `responses/`

- All LLM API responses saved as JSON
- Useful for debugging and auditing
- Timestamps included for tracking

---

## Configuration

### Environment Variables (`.env`)

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=DeepSeek-V3.2-Speciale
AZURE_API_VERSION=2024-02-15-preview

# Model Settings
DEFAULT_MODEL=gpt-4o-mini
STRICT_JSON_MODE=true

# Feature Configuration
SAVE_RESPONSES=true
```

---

## Key Features

### 1. **Intelligent Feature Engineering**

- Automatic data type detection
- Smart feature selection
- Encoding strategy recommendations
- Derived feature suggestions

### 2. **Code Generation**

- Production-ready ML code
- Multiple framework support (scikit-learn, XGBoost)
- Proper train/test splitting
- Model evaluation included

### 3. **Flexibility**

- Multiple LLM models supported
- Domain-specific analysis
- OpenML integration
- Command-line and interactive modes

### 4. **Reliability**

- Robust error handling
- Automatic retry logic
- JSON validation
- Response caching and debugging

### 5. **Extensibility**

- Modular agent architecture
- Easy to add new agents
- Custom domain experts
- Plugin-ready design

---

## Tech Stack

| Component             | Technology             |
| --------------------- | ---------------------- |
| **Language**          | Python 3.9+            |
| **AI/LLM**            | Azure OpenAI API       |
| **Data Processing**   | Pandas, NumPy          |
| **ML Frameworks**     | Scikit-learn, XGBoost  |
| **API Communication** | Requests               |
| **Environment**       | Docker, Docker Compose |
| **Data Validation**   | CAAFE Framework        |

---

## Dependencies

```
scipy>=1.10.0          # Scientific computing
pandas>=3.0.3          # Data manipulation
python-dotenv>=1.0.0   # Environment configuration
requests>=2.31.0       # HTTP requests
numpy>=2.4.6           # Numerical computing
scikit-learn>=1.9.0    # ML algorithms
xgboost>=3.2.0         # Gradient boosting
```

---

## Use Cases

### 1. **Rapid Prototyping**

Get a working ML model in minutes instead of hours

### 2. **Feature Engineering Assistance**

Get AI-recommended features and transformations

### 3. **Code Generation**

Generate boilerplate ML code automatically

### 4. **Data Exploration**

Understand dataset structure and quality issues

### 5. **Learning Resource**

Study generated code to learn ML best practices

### 6. **Domain-Specific Analysis**

Add business context to data science workflows

---

## Workflow Example: Customer Churn Prediction

**Input**: `data/customer_churn.csv`

```
Customer_ID, Age, Tenure, Monthly_Charges, Total_Charges, Churn
1, 25, 2, 55.5, 1800, 0
2, 45, 48, 120.0, 6000, 1
...
```

**AI-Agent Processing**:

1. **Feature Engineer** analyzes structure
   - Keep: Age, Tenure, Monthly_Charges, Total_Charges
   - Scale: Age, Monthly_Charges, Total_Charges
   - Drop: Customer_ID (identifier)
   - Create: Avg_Monthly_Charges = Total/Tenure

2. **Modeler** recommends approach
   - Problem: Binary Classification
   - Suggested: XGBoost, RandomForest
   - Generates complete training code

3. **Domain Expert** (if "telecom" specified)
   - Identifies churn-specific metrics
   - Highlights data quality issues
   - Suggests business-relevant features

4. **Output**: `generated_code/customer_churn_model.py`
   - Ready to train and evaluate
   - Can predict churn immediately

---

## Agent Communication Flow

```
User Input (CSV File)
    ↓
[Main] Reads file and extracts structure
    ↓
[Manager] Orchestrates analysis
    ├─→ [FeatureEngineer] → JSON: {features, scale, encode, drop}
    ├─→ [Modeler] → JSON: {code, model_type, params}
    ├─→ [DomainExpert] → JSON: {insights, metrics, questions}
    └─→ [OpenML] → JSON: {similar_datasets, insights}
    ↓
[Manager] Aggregates results
    ↓
Output: Console display + Generated Python code
```

---

## Quality & Validation

### Feature Validation

- CAAFE framework validates recommendations
- Checks for data type consistency
- Ensures column names match actual data
- Validates encoding strategies

### Error Handling

- Automatic retry logic for LLM failures
- Fallback mechanisms for invalid JSON
- Detailed error messages and logging
- Response debugging via saved JSON files

### Code Quality

- Generated code follows ML best practices
- Includes error handling
- Reproducible results with random seeds
- Model serialization included

---

## Execution Summary

**End-to-End Process:**

1. **Input** → CSV file selection
2. **Parsing** → File structure analysis
3. **Analysis** → 4 parallel agent analyses
4. **Generation** → Python code creation
5. **Output** → Console display + saved files
6. **Result** → Ready-to-run ML pipeline

**Typical Execution Time**: 30-60 seconds per file

---

## Key Advantages

**Time-Saving**: Automates hours of manual work
**Consistency**: Repeatable, standardized workflows
**Quality**: AI-powered best practices built-in
**Flexibility**: Works with any CSV dataset
**Learning**: Great for understanding ML pipelines
**Scalable**: Handles small to medium datasets
**Transparent**: View all generated code and reasoning

---

## Notes for Presenters

### Demo Flow:

1. Show directory structure and sample CSV files
2. Run `python main.py` and select a dataset
3. Demonstrate interactive analysis in real-time
4. Show generated Python code in editor
5. Explain how each agent contributes to the pipeline

### Talking Points:

- "This tool removes the tedious parts of data science"
- "Every generated line of code is AI-validated"
- "You can learn from the generated code"
- "Perfect for rapid prototyping and MVPs"
- "Extensible for custom domains and workflows"

---

## Additional Resources

- **GitHub**: https://github.com/Lord-Sinq/AI-Agent
- **Azure OpenAI Docs**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **OpenML**: https://www.openml.org/
- **Scikit-learn**: https://scikit-learn.org/

---

## Support & Questions

For issues or questions:

1. Check Docker logs: `docker logs ai-agent`
2. Review saved responses in `responses/` directory
3. Check `.env` configuration
4. Verify Azure OpenAI credentials

---

**Created**: 2026-06-10
**Project Status**: Active
**License**: See LICENSE file
**Maintainer**: Lord-Sinq
