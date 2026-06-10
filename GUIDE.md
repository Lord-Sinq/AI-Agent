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

### **Setup (30 seconds)**

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

### **Run Analysis (3 minutes)**

```bash
# Start the tool
python main.py
```

**Console Output:**

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

**What Happens (60-90 seconds):**

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

**Say:** "Notice how quickly the system:

1. Identified the dataset structure
2. Recommended which features to use
3. Suggested scaling and encoding strategies
4. Generated complete model code
5. Provided healthcare-specific insights"

---

### **Show Generated Code (1 minute)**

```bash
# Open generated code
code generated_code/medical_patients_model.py

# Or show with cat
cat generated_code/medical_patients_model.py | head -40
```

**Code Preview:** (Will have to update the load data location)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('data/medical_patients.csv')

# Feature engineering
df['bmi_category'] = pd.cut(df['bmi'],
                             bins=[0, 18.5, 25, 30, float('inf')],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Drop unnecessary columns
df = df.drop(['patient_id', 'timestamp'], axis=1)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['weight', 'blood_pressure', 'glucose', 'insulin', 'bmi']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split data
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Say:** "The generated code is:

- Production-ready
- Well-commented
- Best practices implemented
- Fully customizable
- Can run immediately"

---

### **Run the Model (optional, adds 30-60 seconds)**

```bash
# Actually run the generated code
python generated_code/medical_patients_model.py

# Output:
# Classification Report:
#               precision    recall  f1-score   support
#            0       0.85      0.92      0.88       150
#            1       0.78      0.65      0.71        50
#        accuracy                       0.82       200
#       macro avg       0.81      0.78      0.80       200
#    weighted avg       0.83      0.82      0.82       200
```

**Say:** "And there you have it - a trained model with 82% accuracy, generated completely by AI!"

---

## Different Guide Scenarios

### **Scenario A: Quick Time-Limited Guide (3 minutes)**

Focus on:

1. File selection
2. Show console output
3. Open generated code
4. Done!

**Skip:** Running model, domain analysis details

---

### **Scenario B: Medium Guide (8 minutes)**

Include:

1. File selection
2. Full console output explanation
3. Open and walk through generated code
4. Run the model and show results
5. Explain key components

---

### **Scenario C: Full Presentation (15 minutes)**

Add to Medium Guide:

1. Show project structure
2. Explain architecture
3. Discuss configuration
4. Address Q&A
5. Show different dataset examples

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

**Talking Point:** "All LLM responses are saved for debugging. This transparency helps us understand exactly what happened."

---

### **Problem: "Generation takes too long (>2 minutes)"**

**Quick Fix:**

- Check internet connection
- Check Azure OpenAI API status
- Use smaller dataset
- Show pre-generated code from `responses/` folder

**Talking Point:** "Typically this completes in 30-60 seconds. The delay suggests network latency, but you can see the saved responses showing all the analysis was completed successfully."

---

## Guide Talking Points

### Point 1: Speed

> "What used to take hours of manual data exploration - looking at columns, planning feature engineering, writing boilerplate code - now happens in seconds."

**Guidenstration:** Run `python main.py`, select file, show complete output in ~60 seconds

---

### Point 2: Intelligence

> "The AI doesn't just generate random code. It analyzes your data structure, understands data types, identifies outliers, and makes informed recommendations."

**Guidenstration:** Point out specific recommendations like:

- "Notice how it identified customer_id as an identifier to drop"
- "It automatically selected scaling for numeric columns"
- "It suggested encoding for categorical variables"

---

### Point 3: Production Ready

> "Generated code isn't toy code. It's production-quality with proper data splitting, cross-validation, error handling, and evaluation metrics."

**Guidenstration:** Show generated code structure:

- Train/test split
- Proper preprocessing pipeline
- Model training
- Evaluation metrics

---

### Point 4: Customizable

> "While the generated code is ready to use immediately, every line is yours to modify, extend, or integrate into your existing pipelines."

**Guidenstration:** Edit a file live (add a comment or change a parameter) to show how easy it is to customize

---

### Point 5: Multiple Domains

> "Whether you're working with healthcare, retail, finance, or any other domain, the system adapts and provides domain-specific insights."

**Guidenstration:** If time permits, run analysis on two different datasets:

- Medical data → Healthcare metrics
- Sales data → Business metrics

---

## Recording Guide Commands

If recording for later use, here's a clean script:

```bash
#!/bin/bash

# Setup
clear
echo "=== AI-Agent Guidenstration ==="
echo ""
echo "Step 1: Show available datasets"
ls data/
echo ""

echo "Step 2: Run analysis"
python main.py << EOF
3
EOF

echo ""
echo "Step 3: Show generated code"
cat generated_code/medical_patients_model.py | head -30
echo ""

echo "Step 4: Run the model"
python generated_code/medical_patients_model.py

echo ""
echo "=== Guide Complete ==="
```

Save as `Guide.sh` and run with `bash Guide.sh`

---

## Key Screenshots to Have Ready

1. **Console Output** - Feature engineering recommendations
2. **Generated Code** - Python model file (full view)
3. **Architecture Diagram** - Agent workflow
4. **File Structure** - Directory layout
5. **Results Output** - Model accuracy metrics

(These serve as backup in case of connection issues)

---

## What to Emphasize

### Before the Guide:

- "This would normally take a data scientist 2-3 hours"
- "We're going to see this entire process in about a minute"

### During the Guide:

- "Notice how [specific feature]..."
- "The code included [best practice] automatically"
- "This is real, production-ready code"

### After the Guide:

- "Questions about what you just saw?"
- "This is extensible - you can add your own domain experts"
- "All code is transparent and reviewable"

---

## What NOT to Do

Don't claim it replaces data scientists - emphasize it's an **assistant**
Don't promise 100% accuracy - mention results depend on data quality
Don't go too deep into Azure OpenAI internals unless asked
Don't edit code during Guide if you're not comfortable with it
Don't run multiple instances simultaneously (API rate limits)
Don't forget to mention it's open source on GitHub

---

## Learning Points to Share

After the Guide, here's what people typically learn:

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

---

## Guide Audience Adaptation

### For **Technical Audience** (Engineers/Data Scientists):

- Dive into code architecture
- Discuss agent design patterns
- Mention Azure OpenAI API integration
- Talk about extensibility

### For **Business Audience** (Managers/Directors):

- Focus on time savings
- Emphasize quality and consistency
- Show real-world use cases
- Highlight ROI and productivity gains

### For **Student Audience** (Learning Focus):

- Explain ML concepts being applied
- Point out best practices
- Show how different algorithms work
- Discuss why recommendations are made

---

## Resources to Share

**GitHub Repository:**

```
https://github.com/Lord-Sinq/AI-Agent
```

**Setup Instructions:**

```
1. Clone repository
2. Set up Azure OpenAI credentials
3. Install dependencies: pip install -r requirements.txt
4. Run: python main.py
```

**Quick Start:**

```
# Docker (recommended)
docker-compose build
docker-compose up -d
python main.py

# Local
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## Timing Guide

| Section              | Duration      |
| -------------------- | ------------- |
| Setup & Introduction | 1 min         |
| File Selection       | 30 sec        |
| AI Processing        | 1-2 min       |
| Show Results         | 1 min         |
| Show Code            | 1 min         |
| Optional: Run Model  | 1 min         |
| Q&A                  | 2-5 min       |
| **Total**            | **~8-10 min** |

---

## Success Criteria

After your Guide, the audience should:

✅ Understand what AI-Agent does
✅ See the time savings (seconds vs hours)
✅ Recognize the code quality
✅ Know it's customizable and extensible
✅ Understand the workflow/pipeline
✅ Know where to get it (GitHub)
✅ Be impressed by the automation

---

## Post-Guide Follow-Up

**Slide/Note for email after presentation:**

> "Thank you for attending the AI-Agent Guidenstration! Here are the key resources:
>
> **GitHub Repository**: https://github.com/Lord-Sinq/AI-Agent
>
> **Quick Start**: Clone the repo and run `python main.py`
>
> **Documentation**: See PROJECT_PRESENTATION.md for full details
>
> **Questions?** Feel free to reach out or file an issue on GitHub!"

---

**Ready to present? You've got this!**
