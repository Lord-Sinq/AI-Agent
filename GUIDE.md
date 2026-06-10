# AI-Agent: Quick Guide

### Pre-Guide Checklist

- [ ] Azure OpenAI credentials configured in `.env`
- [ ] Python environment activated
- [ ] Docker running (if using Docker setup)
- [ ] Test CSV files present in `data/` folder
- [ ] Generated code directory exists
- [ ] Backup screenshots ready (just in case)
- [ ] Terminal/IDE open and ready
- [ ] Internet connection stable

---

## How to Run

### Option A: With Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/Lord-Sinq/AI-Agent
cd AI-Agent

# Setup environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Build and run
docker-compose build
docker-compose up -d

# Run application
python main.py
```

### Option B: Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with credentials

# Run
python main.py
```

---

## 🚀 5-Minute Guide Script

### **Setup**

```bash
# Show current directory
ls -la

# Show available data files
ls data/

# Show the main entry point
cat main.py | head -20
```

**Say:** "We have several CSV datasets ready to analyze."

---

### **Run Analysis**

```bash
# Start the tool
python main.py
```

**Console Output: Example**

```
Available files in 'data':
  1. employees.csv
  2. customer_churn.csv
  3. medical_patients.csv
  4. sales_transactions.csv
  5. real_estate.csv
  6. weather_data.csv

Select a file (1-6): 3
```

**Say:** "Let's analyze the medical patients dataset to predict patient outcomes."

**User Input:** Type `3`

**What Happens:**

- System reads the CSV
- Feature Engineer agent analyzes columns
- Modeling agent generates code
- Domain Expert provides healthcare insights
- OpenML looks for similar datasets
- Results displayed and code generated

**Console Output Sample:**

```
[INFO] Reading file: data/medical_patients.csv
[INFO] Extracting structure...

=== FEATURE ENGINEERING RECOMMENDATIONS ===
Features to keep: [age, weight, blood_pressure, glucose, insulin, bmi]
Features to scale: [weight, blood_pressure, glucose, insulin, bmi]
Features to encode: {}
Features to drop: [patient_id, timestamp]
Derived features: ['bmi_category']
Feature metadata: {"reason": "categorical BMI for easier interpretation"}

=== MODEL RECOMMENDATIONS ===
Problem Type: Classification (predict diabetes)
Suggested Models: [XGBoost, RandomForest, LogisticRegression]
Code generated: generated_code/medical_patients_model.py

=== DOMAIN INSIGHTS ===
Domain: Healthcare
Key metrics: [sensitivity, specificity, auc_roc, precision, recall]
Data quality issues: ["Missing values in insulin", "Outliers in age"]
Business questions:
  - Which patients are at highest risk?
  - How does BMI correlate with outcomes?
  - Are there seasonal patterns?

Processing complete in 45 seconds!
```

---

## Troubleshooting During Guide

### **Problem: "Azure OpenAI connection error"**

**Quick Fix:**

```bash
# Check .env configuration
cat .env

# Verify credentials are set
echo $AZURE_OPENAI_API_KEY  # Should show value, not empty

# Try listing deployments
python main.py --list-deployments
```

**Talking Point:** "This would typically indicate missing Azure OpenAI credentials. In a real environment, we'd verify the API keys are current."

---

### **Problem: "No files found in data directory"**

**Quick Fix:**

```bash
# Show files exist
ls -la data/

# Create a test file if needed
cp data/employees.csv data/test.csv
```

---

### **Problem: "JSON parsing error"**

**Quick Fix:**

```bash
# Check saved responses for debugging
ls responses/
cat responses/latest_response.json  # Review what went wrong
```

---

### **Problem: "Generation takes too long (>2 minutes)"**

**Quick Fix:**

- Check internet connection
- Check Azure OpenAI API status
- Use smaller dataset
- Show pre-generated code from `responses/` folder

---

## Learning Points to Share

What people can learn:

1. **Feature Engineering Importance**
   - Why certain features are selected
   - Scaling vs encoding strategies
   - Derived feature creation

2. **ML Pipeline Best Practices**
   - Proper train/test splitting
   - Preprocessing strategies
   - Model evaluation metrics

3. **Code Quality**
   - How professional ML code is structured
   - Important libraries and patterns
   - Error handling and validation

4. **AI Capabilities**
   - What modern LLMs can do with domain knowledge
   - JSON structured output
   - Intelligent analysis of data
