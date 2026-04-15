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

(Before deploying the Streamlit application, the trained XGBoost model was independently tested using a dedicated validation script (model_test.py) to ensure consistency and correctness of predictions.

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

1. Categorical features are encoded using the saved `encoders.pkl`
2. Feature order is aligned using `feature_order.pkl`
3. Numerical features are scaled using `scaler.pkl`
4. The trained `XGBoost` model generates churn probability
5. The saved threshold (`threshold.pkl`) is applied
6. The final churn prediction and confidence score are displayed in the UI

This ensures prediction consistency between training and deployment environments.

---

## Example Usage Flow

1. Enter customer profile details (contract, tenure, payment method, services, etc.)
2. Click **Run Churn Prediction**
3. The system outputs:

   - Churn Probability Score
   - Model Confidence Score
   - Risk Classification (Likely to Stay / Likely to Churn)
   - Retention Recommendation Message

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

- churn_prob → model output
- risk_level → Low / Medium / High
- reasons → churn drivers
- strategies → retrieved knowledge
- sources → references
- final_output → structured JSON response

This ensures modular, explainable, and scalable AI behavior.

## 12. Project Structure

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
├── Report/                              # Final report
│   └── Telecom_Churn_...pdf
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


## 12.1 Retrieval-Augmented Generation (RAG)

The system uses RAG to ensure grounded and context-aware recommendations.

### Process:

- Knowledge is stored in `retention_knowledge.json`
- Converted into documents
- Embedded using MiniLM model
- Stored in FAISS vector database
- Retrieved using similarity search

### Benefits:

- Reduces hallucination
- Improves accuracy
- Provides source-based recommendations

## 12.2 Prompt Engineering & Safety

The LLM is controlled using strict instructions:

- Use only retrieved strategies
- Do not generate new information
- Return JSON output
- Handle missing recommendations safely

This ensures reliable and explainable outputs.

## 13. Technology Stack

| Component | Technology |
|------------|------------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Models | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, LabelEncoder |
| Deployment | Streamlit |
| Model Storage | Joblib |
| Agent Framework | LangGraph |
| Vector Database | FAISS |
| Embeddings | HuggingFace MiniLM |
| LLM | Groq (LLaMA 3.3) |

---

## 14. Milestone Deliverables

### Milestone 1 (Completed)

- Business understanding
- EDA and feature engineering
- Model comparison
- Threshold optimization
- Working local Streamlit application
- Model artifacts saved
- Performance evaluation report

### Milestone 2 (Completed)

- Agentic AI workflow using LangGraph
- RAG-based retrieval using FAISS
- Structured JSON output generation
- Prompt engineering for hallucination control
- Explainable recommendations with sources
- Integrated ML + AI pipeline

---

## 15. Future Improvements

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

## 16. Conclusion

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
