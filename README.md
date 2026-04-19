# Project 5: Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

---

## 1. Project Overview

This project presents the design and implementation of an AI-driven customer analytics system that predicts customer churn and evolves into an agentic AI-powered retention strategist.

The system is divided into two milestones:

### Milestone 1: ML-Based Churn Prediction

A classical machine learning pipeline was developed to predict whether a telecom customer is likely to churn based on:

- Service usage
- Contract type
- Billing details
- Demographic attributes

The goal is to identify high-risk customers early so that proactive retention strategies can be applied.

### Milestone 2: Agentic Retention Intelligence System (Implemented)

The system has been extended into a fully functional Agentic AI pipeline that:

- Performs churn risk reasoning using model outputs
- Uses RAG (FAISS + embeddings) to retrieve relevant retention strategies
- Implements a multi-step agent workflow using LangGraph
- Generates structured, explainable retention reports using an LLM
- Ensures grounded outputs using anti-hallucination prompting

This transforms the system from prediction → reasoning → action.

## 2. Problem Statement

Customer churn is a critical issue for subscription-based businesses because losing existing customers directly impacts revenue and long-term growth. Acquiring new customers is significantly more expensive than retaining current ones.

The objective of this project is to:

- Predict customers who are likely to churn
- Identify key factors contributing to churn
- Support proactive and data-driven retention strategies
- Transition from predictive analytics to intelligent AI-assisted intervention

---
## Business Impact

Customer churn directly affects revenue, customer lifetime value, and growth.

- Reducing churn by even 5% can significantly increase profits  
- Early detection allows proactive intervention  
- AI-driven retention strategies improve decision-making  

This system helps businesses move from reactive to proactive retention.

## 3. Dataset Information

Dataset: Telco Customer Churn Dataset  
Total Records: 7043  
Features: 20 input variables + 1 target variable

The dataset contained 11 missing values in the `TotalCharges` column. 
These entries corresponded to customers with zero tenure. 
Since they represented a very small portion of the dataset, these rows were removed to maintain data consistency.

Target Variable:
- `Churn` (Yes/No)

Feature Categories:

1. Customer Demographics
   - Gender
   - SeniorCitizen
   - Partner
   - Dependents

2. Services Subscribed
   - PhoneService
   - MultipleLines
   - InternetService
   - OnlineSecurity
   - OnlineBackup
   - DeviceProtection
   - TechSupport
   - StreamingTV
   - StreamingMovies

3. Account Information
   - Tenure
   - Contract
   - PaperlessBilling
   - PaymentMethod
   - MonthlyCharges
   - TotalCharges

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

- Removed irrelevant column: `customerID`
- Converted `TotalCharges` to numeric and handled missing values
- Dropped 11 rows with zero tenure
Categorical features were label encoded using `LabelEncoder` and the encoders were saved using joblib for deployment consistency.

Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler`.
Class imbalance was handled in the XGBoost model using the `scale_pos_weight` parameter to improve recall for churn customers.

A stratified train-test split was used to maintain class distribution.

Preprocessing artifacts saved for deployment consistency:
- encoders.pkl
- scaler.pkl
- feature_order.pkl

---

## 5. Key Churn Drivers & Data Insights
<img width="814" height="519" alt="Screenshot 2026-03-01 at 20 23 56" src="https://github.com/user-attachments/assets/73b20358-c436-4b6f-8f25-36642a904f27" />


# 6. Model Performance
## 6.1 Classification Metrics

Three models were evaluated using a stratified train-test split (70:30).

### 1. Logistic Regression

•⁠  ⁠Accuracy: 0.80  
•⁠  ⁠Recall (Churn Class): 0.56  
•⁠  ⁠Precision (Churn Class): 0.64  
•⁠  ⁠F1-Score (Churn Class): 0.60  

Logistic Regression achieved the highest overall accuracy (80%). However, recall for churn customers was moderate (56%), meaning a significant number of churn cases were missed.

---

### 2. Decision Tree

•⁠  ⁠Accuracy: 0.78  
•⁠  ⁠Recall (Churn Class): 0.38  
•⁠  ⁠Precision (Churn Class): 0.64  
•⁠  ⁠F1-Score (Churn Class): 0.48  

Although Decision Tree achieved competitive accuracy (78%), it performed poorly in identifying churn customers, with recall dropping to 38%. This makes it less suitable for churn detection where identifying at-risk customers is critical.

---

### 3. XGBoost (Default Threshold = 0.5)

•⁠  ⁠Accuracy: 0.75  
•⁠  ⁠Recall (Churn Class): 0.75  
•⁠  ⁠Precision (Churn Class): 0.52  
•⁠  ⁠F1-Score (Churn Class): 0.62  

XGBoost significantly improved recall for churn customers (75%), meaning it correctly identified a larger proportion of customers likely to leave. However, overall accuracy was slightly lower compared to Logistic Regression.

---

### 4. XGBoost (Adjusted Threshold = 0.4)

•⁠  ⁠Accuracy: 0.72  
•⁠  ⁠Recall (Churn Class): 0.81  
•⁠  ⁠Precision (Churn Class): 0.48  
•⁠  ⁠F1-Score (Churn Class): 0.60  

After lowering the classification threshold from 0.5 to 0.4, recall improved further to 81%. This ensures that most high-risk customers are detected, even though overall accuracy decreases slightly.

---

## 6.2 Final Model Selection

XGBoost with threshold adjustment (0.4) was selected as the final model.

Although Logistic Regression achieved higher accuracy, churn prediction prioritizes recall over accuracy. Missing a churn customer has higher business cost than incorrectly flagging a loyal one.

The final model achieves:

•⁠  ⁠High churn detection rate (81% recall)
•⁠  ⁠Improved identification of high-risk customers
•⁠  ⁠Business-aligned performance optimization

## 6.3 Error-Based Metrics (MAE, RMSE)

Although churn prediction is a classification task, additional error metrics were computed for comparative evaluation.
Error-Based Metrics (Binary Representation)

1. Logistic Regression
• MAE: 0.20
• RMSE: 0.447
2. Decision Tree
• MAE: 0.222
• RMSE: 0.471
3. XGBoost (Threshold 0.5)
• MAE: 0.248
• RMSE: 0.498
4. XGBoost (Threshold 0.4 – Final Model)
• MAE: 0.282
• RMSE: 0.531

Interpretation:

MAE represents the proportion of incorrect predictions.
RMSE penalizes larger prediction errors more strongly.

The final model shows slightly higher MAE due to recall prioritization, which intentionally increases false positives in order to detect more churn customers.

## 6.4 ROC-AUC Analysis

1. XGBoost ROC-AUC: 0.83
2. The ROC-AUC score measures the model’s ability to distinguish between churn and non-churn customers across all classification thresholds.
3. A value of 0.83 indicates strong classification capability and confirms that the model effectively separates high-risk and low-risk customers.
4. This further validates the robustness of the selected XGBoost model.

## 6.5 Threshold Optimization

The default classification threshold (0.5) was reduced to 0.4 to improve recall for churn customers.

Instead of:

```
prediction = model.predict(X_test)
```

The system uses:

```
probabilities = model.predict_proba(X_test)[:,1]
prediction = (probabilities > 0.4).astype(int)
```

This improves churn detection performance.

Threshold value is saved in:
```
threshold.pkl
```

---

## 7. Key Churn Drivers Identified

EDA revealed strong predictors of churn:

- Month-to-month contracts increase churn
- Electronic check payment method correlates with higher churn
- Low tenure customers churn more frequently
- Lack of tech support increases churn probability
- High monthly charges increase churn risk
- Customers without online security or backup services churn more

Gender showed minimal impact on churn.

---

## 8. Streamlit Deployment

A production-ready Streamlit application was developed.

Features:

- Interactive customer profile input
- Real-time churn probability prediction
- Custom threshold-based classification
- Styled UI with responsive layout
- Probability visualization with progress bar
- Retention recommendation messaging

The application correctly:

- Applies saved encoders
- Reorders features to match training
- Scales numerical columns
- Uses the saved model and threshold

---

## 9. Local Model Testing & Validation

Before deploying the Streamlit application, the trained XGBoost model was independently tested using a dedicated validation script (model_test.py) to ensure consistency and correctness of predictions.

The testing process verified:

1. Successful loading of saved model artifacts
2. Proper encoder alignment
3. Correct feature ordering
4. Numerical feature scaling
5. Threshold-based classification logic
6. Final churn prediction output

Example local test output:

<img width="838" height="370" alt="image" src="https://github.com/user-attachments/assets/07d625e6-4d6c-4519-821e-66bbe7dbaaf0" />

---

---

## 10. How to Run & Use This Project

This project can be executed locally using Streamlit.

Follow the steps below to run the application on your system.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/connectwithvanshika/customer_churn_prediction.git
cd customer_churn_prediction
```

---

### Step 2: Install Required Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

---

### Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your browser at:

```
http://localhost:8501
```

---

## How the System Works Internally

When a user enters customer details in the UI and clicks **Run Churn Prediction**, the system performs the following steps:

**Milestone 1 (ML Prediction):**

1. Categorical features are encoded using the saved `encoders.pkl`
2. Feature order is aligned using `feature_order.pkl`
3. Numerical features are scaled using `scaler.pkl`
4. The trained `XGBoost` model generates churn probability
5. The saved threshold (`threshold.pkl`) is applied

**Milestone 2 (Agentic AI):**

6. Churn probability is passed to the LangGraph agent
7. Risk Node classifies risk level and identifies churn reasons
8. Retrieval Node queries the FAISS vector database for matching strategies
9. Planning Node sends context to Groq LLM (LLaMA 3.3-70B) for structured output
10. Final JSON output is parsed and displayed in the UI

This ensures prediction consistency between training and deployment environments.

---

## Example Usage Flow

1. Enter customer profile details (contract, tenure, payment method, services, etc.)
2. Click **Run Churn Prediction**
3. The system outputs:

   - Churn Probability Score with gauge visualization
   - Risk Classification (Low / Medium / High)
   - Top Risk Factors Identified
   - AI-Generated Retention Recommendations (from RAG + LLM)
   - Customer Profile Summary Table
   - Source-backed strategy references

This allows proactive and data-driven retention decisions.

---

## Using the Model Programmatically (Optional)

You can also use the trained model directly in Python:

```python
import joblib
import numpy as np

# Load saved artifacts
model = joblib.load("final_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
threshold = joblib.load("threshold.pkl")

# Example: pass processed feature array
probability = model.predict_proba(X_sample)[:, 1]
prediction = (probability > threshold).astype(int)

print("Churn Probability:", probability)
print("Final Prediction:", prediction)
```

---

## Deployment-Ready Artifacts

The following artifacts are saved to ensure reproducibility:

- `final_churn_model.pkl`
- `scaler.pkl`
- `encoders.pkl`
- `threshold.pkl`
- `feature_order.pkl`

These guarantee that the deployed application produces predictions identical to the training environment.

---


## 11. Project Architecture

### Milestone 1 Architecture

User Input → Encoding → Feature Ordering → Scaling → XGBoost Model → Probability → Threshold Logic → UI Output

### Milestone 2 Architecture (Implemented)

User Input → Preprocessing → ML Model → Risk Node → RAG (FAISS) → LLM (Groq) → Structured Output

---

## Architecture Diagram
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0b241f50-9dac-434b-b97b-793e25cc6a54" />


## 11.1 Agentic AI System Architecture (Milestone 2)

The system integrates machine learning, retrieval systems, and LLM-based reasoning.

### Pipeline Flow:

User Input → ML Model → Risk Analysis → Retrieval (RAG) → Planning (LLM) → Structured Output

### Components:

1. ML Layer (XGBoost)
   - Predicts churn probability

2. Risk Analysis Node
   - Converts probability into risk levels
   - Identifies churn drivers (low tenure, high charges)

3. Retrieval Layer (RAG)
   - FAISS vector database
   - Embeddings: all-MiniLM-L6-v2
   - Retrieves relevant strategies

4. Planning Node (LLM - Groq)
   - Generates structured JSON output
   - Uses strict prompt rules

5. Output Layer
   - Risk summary
   - Recommendations
   - Sources
   - Disclaimer

## 11.2 Agent Workflow (LangGraph)

The system uses LangGraph to orchestrate a multi-step AI workflow.

### Workflow:

risk_node → retrieval_node → planning_node

### Agent State (AgentState):

```python
class AgentState(TypedDict):
    churn_prob: float       # ML model output (0.0 to 1.0)
    tenure: int             # Customer tenure in months
    monthly: float          # Monthly charges in USD
    risk_level: str         # Low / Medium / High
    reasons: List[str]      # Identified churn drivers
    strategies: List[str]   # Retrieved retention strategies from RAG
    sources: List[str]      # Source references from knowledge base
    final_output: str       # Structured JSON response from LLM
```

This ensures modular, explainable, and scalable AI behavior.

---

## 11.3 Detailed LangGraph Node Implementation

The LangGraph workflow consists of three sequential nodes. Each node has a specific responsibility and passes an updated state to the next node.

### Node 1: Risk Analysis Node (`risk_node`)

**Purpose:** Converts the raw churn probability into an interpretable risk level and identifies the primary reasons driving the risk.

**Logic:**

```python
def risk_node(state: AgentState):
    prob = state["churn_prob"]

    # Risk classification thresholds
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # Identify churn drivers from customer profile
    reasons = []
    if state["tenure"] < 6:
        reasons.append("low_tenure")
    if state["monthly"] > 80:
        reasons.append("high_charges")
    if not reasons:
        reasons.append("general")

    return {**state, "risk_level": risk, "reasons": reasons}
```

**Input:** `churn_prob`, `tenure`, `monthly`  
**Output:** `risk_level` (Low/Medium/High), `reasons` (list of churn drivers)

**Design Decision:** Threshold values (0.7 for High, 0.4 for Medium) align with the model's optimized classification threshold to maintain consistency between ML output and agent reasoning.

---

### Node 2: Strategy Retrieval Node (`retrieval_node`)

**Purpose:** Uses the identified churn reasons as a semantic query to retrieve the most relevant retention strategies from the FAISS vector database.

**Logic:**

```python
def retrieval_node(state: AgentState):
    # Build query from churn drivers
    query = " ".join(state["reasons"])
    
    # Semantic similarity search over the knowledge base
    results = vectorstore.similarity_search(query, k=3)
    
    strategies = []
    sources = []

    for doc in results:
        strategies.append(doc.page_content)
        sources.append(doc.metadata["source"])

    return {
        **state,
        "strategies": list(set(strategies)),  # Deduplicate
        "sources": list(set(sources))          # Deduplicate
    }
```

**Input:** `reasons` (churn drivers from Node 1)  
**Output:** `strategies` (retrieved knowledge), `sources` (references)

**Design Decision:** Deduplication (`list(set(...))`) prevents the LLM from receiving redundant information, which reduces prompt length and improves output quality.

---

### Node 3: Planning & Recommendation Node (`planning_node`)

**Purpose:** Uses the retrieved strategies and customer risk profile to generate a structured JSON recommendation report via the Groq LLM (LLaMA 3.3-70B).

**Prompt Engineering (Anti-Hallucination Rules):**

```python
prompt = f"""
You are an AI Customer Retention Strategist.

Customer churn probability: {state['churn_prob']}
Risk level: {state['risk_level']}
Reasons: {state['reasons']}

Retrieved Strategies: {state['strategies']}
Sources: {state['sources']}

IMPORTANT RULES:
- Use ONLY the provided strategies and sources
- Do NOT generate new strategies
- If no relevant strategy, say "No recommendation found"

RETURN STRICT JSON:
{{
  "risk_summary": "Explain churn risk in detail (2-3 lines with reasoning)",
  "recommendations": [
    "Detailed action 1 with explanation",
    "Detailed action 2 with explanation"
  ],
  "sources": ["source1", "source2"],
  "business_impact": "Explain what happens if no action is taken (2 lines)",
  "disclaimer": "This prediction is probabilistic and may not guarantee actual churn."
}}

ONLY return JSON. No extra text.
"""
```

**Output Structure:**

| Field | Description |
|---|---|
| `risk_summary` | 2-3 line explanation of the customer's churn risk |
| `recommendations` | List of specific retention actions |
| `sources` | Knowledge base references used |
| `business_impact` | What happens if no action is taken |
| `disclaimer` | Ethical AI disclosure |

**Fallback Handling:**

```python
try:
    parsed_output = json.loads(raw_output)
except:
    parsed_output = {
        "risk_summary": "Parsing error",
        "recommendations": [],
        "sources": [],
        "disclaimer": "Model output could not be parsed"
    }
```

This ensures the application never crashes due to unexpected LLM output formatting.

---

### LangGraph Graph Construction

```python
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)

# Register nodes
builder.add_node("risk", risk_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("planning", planning_node)

# Define linear workflow
builder.set_entry_point("risk")
builder.add_edge("risk", "retrieval")
builder.add_edge("retrieval", "planning")

# Compile the graph
graph = builder.compile()

# Invoke with customer data
result = graph.invoke({
    "churn_prob": 0.82,
    "tenure": 4,
    "monthly": 95.0
})
```

---

## 12. Retrieval-Augmented Generation (RAG) System

### 12.1 RAG Architecture Overview

The RAG system ensures that all retention strategy recommendations are grounded in a curated domain knowledge base — preventing the LLM from hallucinating advice that has no factual basis.

```
retention_knowledge.json → Document Loader → HuggingFace Embeddings → FAISS Vector Store → Similarity Search → LLM Context
```

### 12.2 Knowledge Base Design (`retention_knowledge.json`)

The knowledge base is a structured JSON file containing expert-curated retention strategies. Each entry follows this schema:

```json
[
  {
    "condition": "Customer has month-to-month contract with high churn probability",
    "strategy": "Offer a discounted upgrade to a 1-year or 2-year contract with added benefits such as free months or service bundles. Long-term contracts reduce churn by 3x.",
    "source": "Telecom Retention Playbook - Contract Upgrade Strategy"
  },
  {
    "condition": "Customer has low tenure (less than 6 months)",
    "strategy": "Initiate an onboarding loyalty program within the first 90 days. Provide personalized check-ins, usage tips, and a first-time loyalty discount.",
    "source": "Customer Success Handbook - Early Lifecycle Retention"
  },
  {
    "condition": "Customer has high monthly charges with no bundled services",
    "strategy": "Offer a tailored service bundle that reduces per-service cost. Highlight total savings. Bundling increases perceived value and reduces price sensitivity.",
    "source": "Revenue Retention Research - Bundling Impact Study"
  }
]
```

**Knowledge Base Stats:**
- Each entry contains: `condition`, `strategy`, `source`
- Conditions map to real churn drivers identified in EDA
- Sources are referenced in the final AI output for transparency

### 12.3 Embedding & Vector Store Setup

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

# Load knowledge base
with open("retention_knowledge.json") as f:
    knowledge = json.load(f)

# Convert to LangChain Document format
docs = []
for item in knowledge:
    content = f"Condition: {item['condition']}\nStrategy: {item['strategy']}"
    docs.append(
        Document(
            page_content=content,
            metadata={
                "source": item["source"],
                "condition": item["condition"]
            }
        )
    )

# Create embeddings using MiniLM (lightweight, free, no API key needed)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS vector store
vectorstore = FAISS.from_documents(docs, embedding_model)
```

**Why `all-MiniLM-L6-v2`?**
- Runs locally — no external API calls
- Fast inference (< 1 second per query)
- 384-dimensional embeddings — ideal for semantic similarity
- Free — fits within the project's free-tier API budget

**Why FAISS?**
- In-memory vector search — no database server required
- Sub-millisecond similarity search for small knowledge bases
- Compatible with LangChain's document retrieval pipeline
- Can be persisted to disk (`vectorstore.save_local(...)`) if needed

---

## 13. ML + Agentic AI Integration Flow

### 13.1 How ML and Agent Communicate

The ML model and the LangGraph agent are decoupled components that communicate through the `prediction_result` dictionary stored in Streamlit's session state.

```
Streamlit Form Submission
        ↓
Customer Dict (19 features)
        ↓
Encode → Reorder → Scale → XGBoost → Probability
        ↓
session_state["prediction_result"] = {"prob": 0.82, "churns": True}
        ↓
Page redirect to Results page
        ↓
graph.invoke({"churn_prob": prob, "tenure": tenure, "monthly": monthly})
        ↓
risk_node → retrieval_node → planning_node
        ↓
Structured JSON rendered in UI
```

### 13.2 Data Flow Between Components

```python
# Step 1: ML Prediction (in page_predict)
prob_arr = models["model"].predict_proba(X_scaled)[0]
prob = float(prob_arr[1])

st.session_state.prediction_result = {
    "prob": prob,
    "churns": prob >= threshold,
}

# Step 2: Agent Invocation (in page_results)
agent_input = {
    "churn_prob": prob,
    "tenure": customer.get("tenure", 0),
    "monthly": customer.get("MonthlyCharges", 0),
}

agent_output = graph.invoke(agent_input)
ai_output = agent_output.get("final_output", {})

# Step 3: Render structured output
recommendations = ai_output.get("recommendations", [])
risk_summary = ai_output.get("risk_summary", "")
business_impact = ai_output.get("business_impact", "")
```

### 13.3 Fallback Strategy

If the ML model artifacts are unavailable (e.g., on a fresh deployment), a rule-based fallback system estimates churn probability:

```python
if prob is None:
    risk = 0.1
    if contract == "Month-to-month": risk += 0.35
    elif contract == "One year":     risk += 0.10
    if payment == "Electronic check": risk += 0.20
    if tech_support == "No":          risk += 0.12
    if internet_service == "Fiber optic": risk += 0.10
    if online_security == "No":       risk += 0.08
    if tenure < 12:                   risk += 0.15
    elif tenure > 48:                 risk -= 0.10
    if monthly > 70:                  risk += 0.08
    prob = min(max(risk, 0.02), 0.98)
```

This ensures the agentic AI system continues to work even if ML model files fail to load.

---

## 14. Prompt Engineering & Hallucination Prevention

### Strategy Used: Constraint-Based Prompting

The LLM is given explicit rules that restrict it from generating information outside the retrieved context:

```
IMPORTANT RULES:
- Use ONLY the provided strategies and sources
- Do NOT generate new strategies
- If no relevant strategy, say "No recommendation found"
```

### Why This Matters

Without these constraints, the LLM could:
- Invent strategies not grounded in the knowledge base
- Reference sources that don't exist
- Generate advice that contradicts telecom domain best practices

### Output Format Enforcement

By requiring strict JSON output with a defined schema, the system:
- Makes responses predictable and parseable
- Enables structured rendering in the UI
- Forces the LLM to organize its reasoning into predefined categories
- Prevents free-form text that could mix fact and hallucination

### Disclaimer Injection

Every response includes a mandatory disclaimer field:

```json
"disclaimer": "This prediction is probabilistic and may not guarantee actual churn."
```

This ensures ethical AI disclosure — users understand the system provides decision support, not deterministic predictions.

---

## 15. Project Structure

The repository is organized to ensure clarity, reproducibility, and deployment readiness.

```
customer_churn_prediction/
│
├── app.py                               # Streamlit application (ML + Agentic AI pipeline)
├── model_test.py                        # Local model validation script
├── requirements.txt                     # Project dependencies
├── README.md                            # Project documentation
├── retention_knowledge.json             # Knowledge base for RAG
├── .env                                 # Environment variables (API keys)
├── .gitignore                           # Version control exclusions
│
├── .streamlit/                          # Streamlit configuration
│   └── config.toml
│
├── Raw_Dataset/                         # Original dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebook_&_otherpkl/                 # Notebook + trained artifacts
│   ├── CUSTOMER_CHURN_PREDICTION_Gen_AI_Project.ipynb
│   ├── final_churn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── threshold.pkl
│   ├── feature_order.pkl
│   ├── retention_knowledge.json         # RAG dataset (used for vector DB)
│   └── milestone2_ra...                 # Additional milestone 2 artifacts
│
├── EDA Insights/                        # EDA visualizations
│   ├── EDA 1.png
│   ├── EDA 2.png
│   └── EDA 3.png
│
├── Milestone 1 and 2 UI/                # UI screenshots for submission
│   ├── Ui_1.png
│   └── Ui_2.png
│
├── Report/                              # Final report and workflow docs
│   ├── Agent_Workflow_Documentation.pdf
│   └── Telecom_Churn_Report.pdf
```


### Structure Overview

- **Application Layer** → `app.py` (ML + RAG + LangGraph + LLM pipeline)
- **Model Artifacts** → `.pkl` files for deployment consistency
- **Validation Layer** → `model_test.py`
- **Knowledge Base (RAG)** → `retention_knowledge.json`
- **Agent Workflow** → LangGraph nodes (risk → retrieval → planning)
- **Configuration** → `.streamlit/config.toml` + `.env`
- **Dataset & Notebook** → Raw data and full ML workflow
- **EDA & UI Assets** → Visual insights and application screenshots
- **Documentation** → README and report files

This structure ensures that the training pipeline, deployment pipeline, and evaluation pipeline remain fully reproducible.

---

---

## 16. Testing & Validation

### 16.1 ML Model Validation (`model_test.py`)

The model was independently validated before deployment using a local test script.

```python
import joblib
import numpy as np
import pandas as pd

# Load all artifacts
model     = joblib.load("notebook_&_otherpkl/final_churn_model.pkl")
scaler    = joblib.load("notebook_&_otherpkl/scaler.pkl")
encoders  = joblib.load("notebook_&_otherpkl/encoders.pkl")
threshold = float(joblib.load("notebook_&_otherpkl/threshold.pkl"))
feature_order = joblib.load("notebook_&_otherpkl/feature_order.pkl")

# Test customer profile
test_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0,
}

# Encode
row = {}
for feat in feature_order:
    val = test_customer.get(feat, 0)
    if feat in encoders:
        val = int(encoders[feat].transform([val])[0])
    elif isinstance(val, str):
        val = 1 if val in ["Yes", "Female"] else 0
    row[feat] = val

X = pd.DataFrame([row])[feature_order]
X_scaled = scaler.transform(X)

prob = model.predict_proba(X_scaled)[0][1]
prediction = int(prob >= threshold)

print(f"Churn Probability : {prob:.4f}")
print(f"Threshold Applied : {threshold}")
print(f"Prediction        : {'CHURN' if prediction == 1 else 'NO CHURN'}")
```

### 16.2 RAG Retrieval Validation

To verify the RAG system returns relevant results:

```python
# Test query
test_query = "low_tenure high_charges"
results = vectorstore.similarity_search(test_query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content : {doc.page_content[:200]}")
    print(f"Source  : {doc.metadata['source']}")
```

Expected behavior: Results should match strategies related to new customer onboarding and high billing concerns.

### 16.3 Agent Pipeline Validation

End-to-end test of the full LangGraph pipeline:

```python
# Test agent with a high-risk customer profile
test_input = {
    "churn_prob": 0.82,
    "tenure": 3,
    "monthly": 95.0,
}

output = graph.invoke(test_input)

print("Risk Level    :", output["risk_level"])
print("Reasons       :", output["reasons"])
print("Strategies    :", len(output["strategies"]), "retrieved")
print("Final Output  :", output["final_output"])
```

**Validation Checklist:**

- [x] Model loads without error
- [x] Encoder transforms all categorical features correctly
- [x] Feature order matches training pipeline
- [x] Scaler normalizes numerical values correctly
- [x] XGBoost outputs probability in [0, 1] range
- [x] Threshold correctly classifies churn vs no-churn
- [x] FAISS retrieves semantically relevant strategies
- [x] LangGraph executes all 3 nodes in order
- [x] Groq API returns valid JSON response
- [x] JSON parser handles malformed responses gracefully
- [x] UI renders all output fields without error

---

## 17. Troubleshooting

### Common Issues & Fixes

**Issue 1: `Error loading model file`**
```
Error loading model file: 'notebook_&_otherpkl/final_churn_model.pkl'
```
**Fix:** Ensure you are running `streamlit run app.py` from the project root directory, not from inside a subdirectory. All `.pkl` files must be in the `notebook_&_otherpkl/` folder relative to `app.py`.

---

**Issue 2: `GROQ_API_KEY not found`**
```
AuthenticationError: No API key provided
```
**Fix:** Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at https://console.groq.com. The `python-dotenv` package loads it automatically on startup.

---

**Issue 3: `FAISS index build fails`**
```
RuntimeError: FAISS index creation failed
```
**Fix:** Ensure `retention_knowledge.json` exists in the project root. The file must be valid JSON with fields: `condition`, `strategy`, `source`. Install dependencies:
```bash
pip install faiss-cpu sentence-transformers
```

---

**Issue 4: `LLM output could not be parsed`**
```
Parsing error — Model output could not be parsed
```
**Fix:** This occurs when Groq's API returns non-JSON text (e.g., markdown code blocks). The system handles this with a fallback, but to reduce frequency: ensure the prompt ends with `ONLY return JSON. No extra text.` — which is already implemented.

---

**Issue 5: `HuggingFace model download fails`**
```
ConnectionError: Unable to fetch model 'all-MiniLM-L6-v2'
```
**Fix:** On first run, the embedding model downloads from HuggingFace (~80MB). Ensure internet connectivity. After first run, it caches locally and works offline.

---

**Issue 6: Application crashes on Streamlit Cloud**
**Fix:** Ensure these files are in your repository root (not just locally):
- `retention_knowledge.json`
- `requirements.txt` (with all dependencies)
- `.streamlit/config.toml`

Add secrets on Streamlit Cloud: Settings → Secrets → Add `GROQ_API_KEY`.

---

## System Input & Output

### Input:
- Customer profile (tenure, services, billing details)

### Output:
- Churn probability score  
- Risk level (Low / Medium / High)  
- AI-generated retention strategies  
- Source-backed recommendations

### Future Agent Enhancement

The workflow can be extended with conditional logic, where:
- Low-risk users skip retrieval step  
- High-risk users trigger full RAG + planning pipeline  

This improves efficiency and makes the system truly agentic.

## System Robustness

- Handles missing or invalid inputs  
- Includes fallback for LLM JSON parsing  
- Prevents hallucination using strict prompts  
- Ensures consistent output structure

## 18. Technology Stack

| Component | Technology |
|---|---|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Models | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, LabelEncoder |
| Deployment | Streamlit |
| Model Storage | Joblib, Pickle |
| Agent Framework | LangGraph |
| Vector Database | FAISS |
| Embeddings | HuggingFace MiniLM (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA 3.3-70B-Versatile) |
| Environment Config | python-dotenv |

---

## 19. Milestone Deliverables

### Milestone 1 (Completed)

- Business understanding
- EDA and feature engineering
- Model comparison
- Threshold optimization
- Working local Streamlit application
- Model artifacts saved
- Performance evaluation report

### Milestone 2 (Completed)

- Agentic AI workflow using LangGraph (3-node pipeline)
- RAG-based retrieval using FAISS + HuggingFace embeddings
- Curated retention knowledge base (`retention_knowledge.json`)
- Structured JSON output generation via Groq LLM
- Prompt engineering for hallucination prevention
- Explainable recommendations with source references
- Integrated ML + Agentic AI pipeline in single Streamlit app
- Analytics Dashboard with 6 interactive Plotly charts
- Full results page with risk factors, gauge chart, and retention actions
- Hosted deployment on Streamlit Community Cloud

---

## 20. Future Improvements

- Add conditional agent flow (skip steps for low-risk users)  
- Integrate SHAP for explainable AI insights  
- Enable real-time data integration via APIs  
- Generate personalized retention strategies  
- Implement feedback loop for continuous learning  
- Expand RAG knowledge base for better recommendations  
- Deploy using Docker & cloud platforms (AWS/GCP)  
- Add monitoring for model performance & drift  
- Improve UI with dashboards & visual insights  
- Secure configuration using environment variables  
- Implement multi-agent architecture for advanced reasoning  
- Perform A/B testing for retention strategies  

---

## 21. Conclusion

This project successfully implements an end-to-end customer churn intelligence system, starting from raw data processing to a fully deployed interactive application.

It goes beyond traditional machine learning by integrating an agentic AI layer, transforming the system from a predictive model into an intelligent decision-support system. By combining ML, RAG, and LLM-based reasoning, the project demonstrates how modern AI systems can move from prediction to actionable business insights.

The system:

- Accurately identifies high-risk customers using a recall-optimized ML model  
- Provides probability-driven insights for better decision-making  
- Incorporates risk analysis to interpret model outputs  
- Uses RAG to retrieve domain-specific retention strategies  
- Generates structured, explainable recommendations using LLMs  
- Ensures transparency through source-backed outputs  
- Is deployed as an interactive and user-friendly application  
- Is designed to be scalable and extensible for real-world business use  

This project highlights the practical application of machine learning and generative AI in solving real-world business problems. It lays a strong foundation for building intelligent, autonomous systems capable of assisting organizations in customer retention and strategic decision-making.

## Deployment

The application is deployed using Streamlit and can be accessed via a public link.

The system integrates:
- ML model inference
- RAG retrieval system
- LLM-based reasoning

All components run in a unified pipeline.

## Dataset Source
IBM Sample Data Sets. Telco Customer Churn Dataset. Kaggle. Available at:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Libraries & Tools
• Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
• Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD.
• McKinney, W. (2010). Data Structures for Statistical Computing in Python. SciPy Conference.
• Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3).
• Plotly Technologies Inc. Plotly Python Graphing Library. https://plotly.com/python/
• Streamlit Inc. Streamlit: The fastest way to build data apps. https://streamlit.io

• LangChain. Framework for developing applications powered by LLMs. https://www.langchain.com/  
• LangGraph. Framework for building stateful, multi-step AI agents.  
• Johnson, J., Douze, M., & Jégou, H. (2017). FAISS: Efficient Similarity Search. Facebook AI Research.  
• HuggingFace. Sentence Transformers & Embeddings Models. https://huggingface.co/  
• Groq Inc. LLM Inference Engine (LLaMA Models). https://groq.com/  

• Python Software Foundation. Python Language Reference. https://www.python.org/  
• Joblib Library. Efficient serialization for ML models.  
• dotenv. Environment variable management for secure configuration.  

## Live Resources
1. Live Colab Notebook - [https://colab.research.google.com/drive/1qUUYKSU4QDwKlGH_H9j1Cr2NyEKXcqIA?usp=sharing]
2. Dataset (Kaggle) - [https://www.kaggle.com/datasets/blastchar/telco-customer-churn]
3. Live Application - [https://customerchurnprediction-2k327gcbblu4dawrhawsit.streamlit.app/]
4. Video Explanation - [https://drive.google.com/drive/u/1/folders/12rKe4vnmFKiYhrQIUKWCfCnlRTNMNwLt]

Developed by Team RetainX AI

Team Members 
1. Vanshika Yadav 
2. ⁠Riya Garg 
3. ⁠Sankalp 
4. ⁠Ronit Singh
