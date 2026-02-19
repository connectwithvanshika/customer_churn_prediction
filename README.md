# Customer Churn Prediction using Machine Learning

This project is a machine learning web application that predicts whether a customer is likely to churn or stay based on customer demographics, services, and billing information. The application is built using Streamlit and demonstrates an end-to-end machine learning workflow including preprocessing, model training, and deployment with an interactive user interface.

## Project Overview
Customer churn prediction helps businesses identify customers who are likely to discontinue their services. By analyzing customer behavior and service usage, this model provides real-time predictions that can help organizations improve customer retention and make data-driven decisions.

## Features
- Real-time customer churn prediction  
- Interactive and user-friendly interface  
- Machine learning model integrated with Streamlit  
- Clean and structured input form  
- Instant prediction results  
- End-to-end ML workflow implementation  

## Machine Learning Workflow
The project follows a complete machine learning pipeline:
1. Data preprocessing and cleaning  
2. Feature encoding  
3. Model training using a classification algorithm  
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

├── .streamlit/  
│   └── config.toml          # Streamlit configuration  
├── app.py                   # Main Streamlit application  
├── churn_model.pkl          # Trained machine learning model  
├── scaler.pkl               # Saved scaler for preprocessing  
├── requirements.txt         # Project dependencies  
└── README.md  

## Installation and Setup

### Clone the repository
git clone https://github.com/connectwithvanshika/customer_churn_prediction.git  
cd project

### Install dependencies
pip install -r requirements.txt  

### Run the application
streamlit run app.py  

The application will open in your browser at:  
http://localhost:8501  

## Input Features Used
The model uses the following inputs:
- Gender  
- Senior Citizen  
- Partner and Dependents  
- Tenure  
- Phone Service and Multiple Lines  
- Internet Service  
- Online Security and Online Backup  
- Device Protection and Tech Support  
- Streaming TV and Movies  
- Contract Type  
- Paperless Billing  
- Payment Method  
- Monthly Charges  
- Total Charges  

## Prediction Output
The application predicts whether a customer is likely to churn or stay. This helps businesses:
- Improve customer retention strategies  
- Identify high-risk customers  
- Make data-driven decisions  

## Future Improvements
- Deploy application on cloud for public access  
- Add model performance metrics and visualizations  
- Integrate real-time database  
- Enhance UI/UX design  
- Add analytics dashboard  

## Team Members
Vanshika Yadav  
Riya Garg  
Ronit Singh  
Sankalp  

GitHub: https://github.com/connectwithvanshika
