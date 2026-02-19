# Customer Churn Prediction App

This is a machine learning web application built using Streamlit that predicts whether a customer is likely to churn or stay based on their demographic and service usage details. The project demonstrates an end-to-end machine learning workflow including preprocessing, model building, and deployment with an interactive user interface.

## Project Overview
Customer churn prediction helps businesses identify customers who are likely to leave their services. This application uses a trained machine learning model to analyze customer data and provide real-time predictions through a simple and clean web interface.

## Features
- Real-time customer churn prediction  
- Interactive and user-friendly interface  
- Machine learning model integrated with Streamlit  
- Clean UI with structured input sections  
- Instant prediction results  

## Machine Learning Workflow
The project follows a complete machine learning pipeline:
1. Data preprocessing and cleaning  
2. Feature encoding  
3. Model training using classification algorithm  
4. Feature scaling using StandardScaler  
5. Model saving using pickle  
6. Deployment using Streamlit  

## Tech Stack
Frontend and UI:
- Streamlit
- Custom CSS

Backend and Machine Learning:
- Python
- NumPy
- Scikit-learn
- Pickle

## Project Structure
customer_churn_prediction/

├── app.py                 # Main Streamlit application  
├── churn_model.pkl        # Trained machine learning model  
├── scaler.pkl             # Saved scaler for preprocessing  
├── .streamlit/config.toml # Streamlit configuration  
└── README.md  

## Installation and Setup

### Clone the repository
git clone https://github.com/connectwithvanshika/customer_churn_prediction.git  
cd customer_churn_prediction  

### Install dependencies
pip install streamlit numpy scikit-learn  

### Run the application
streamlit run app.py  

The application will open in your browser at:  
http://localhost:8501  

## Input Features Used
The model takes the following inputs:
- Gender  
- Senior Citizen  
- Partner and Dependents  
- Tenure  
- Phone and Internet Services  
- Online Security and Backup  
- Device Protection and Tech Support  
- Streaming Services  
- Contract Type  
- Payment Method  
- Monthly Charges  
- Total Charges  

## Prediction Output
The application predicts whether a customer is likely to churn or stay. This can help businesses:
- Improve customer retention strategies  
- Identify high-risk customers  
- Make data-driven decisions  

## Future Improvements
- Deploy the application on Streamlit Cloud  
- Add model accuracy and evaluation metrics  
- Include visual dashboards for insights  
- Connect with real-time database  
- Enhance UI and user experience  

## Team Members
Vanshika Yadav  
Riya Garg  
Ronit Singh  
Sankalp  

