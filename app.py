import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\Hp\Downloads\online_fraud.csv")

# Step 2: Preprocess the data
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Drop 'isFlaggedFraud' column if it exists
if 'isFlaggedFraud' in data.columns:
    data = data.drop('isFlaggedFraud', axis=1)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Add custom HTML and CSS for styling with gradients
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #faaca8, #ddd6f3);
        font-family: Arial, sans-serif;
    }
    .main-container {
        background: linear-gradient(to right, #2c3e50, #bdc3c7);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 36px;
        margin-bottom: 20px;
        background: linear-gradient(to right, #1d2671, #c33764);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .description {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(to right, #faaca8, #ddd6f3);
        color: black;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #c33764, #1d2671);
        color: white;
        transform: scale(1.05);
    }
    .result-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 20px;
    }
    .result-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Step 6: Build the Streamlit app
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Fraud Transaction Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Enter transaction details to predict whether it is fraudulent or not.</p>', unsafe_allow_html=True)

# Input fields for user data
user_input = {}
for col in X.columns:
    if col in label_encoders:  # Categorical columns
        options = list(label_encoders[col].classes_)
        user_input[col] = st.selectbox(f"{col}", options)
    else:  # Numerical columns
        user_input[col] = st.number_input(f"{col}", value=0.0, format="%.2f")

# Convert input to dataframe
input_df = pd.DataFrame([user_input])
for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.markdown('<div class="result-error">This transaction is predicted to be FRAUDULENT.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-success">This transaction is predicted to be NOT FRAUDULENT.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
