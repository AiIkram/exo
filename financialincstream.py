import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("rf_model.pkl")

# Title of the Streamlit App
st.title("Bank Account Prediction App")

# Input fields for the user to provide data
st.header("Provide the following details:")

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
location_type = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=30, step=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=16, max_value=100, step=1)
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married/Living together", "Divorced/Separated", "Widowed", "Single/Never Married"])
education_level = st.selectbox("Education Level", [
    "No formal education", "Primary education", "Secondary education",
    "Vocational/Specialised training", "Tertiary education", "Other/Dont know/RTA"
])
job_type = st.selectbox("Job Type", [
    "Farming and Fishing", "Self employed", "Formally employed Government",
    "Formally employed Private", "Informally employed", "Remittance Dependent",
    "Government Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"
])
relationship_with_head = st.selectbox("Relationship with Head", [
    "Head of Household", "Spouse", "Child", "Parent", "Other relative",
    "Other non-relative"
])

# Button to predict
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        "country": [country],
        "location_type": [location_type],
        "cellphone_access": [cellphone_access],
        "household_size": [household_size],
        "age_of_respondent": [age_of_respondent],
        "gender_of_respondent": [gender_of_respondent],
        "marital_status": [marital_status],
        "education_level": [education_level],
        "job_type": [job_type],
    })

    # Make predictions
    prediction = model.predict(input_data)
    st.write("Prediction:", "Yes" if prediction[0] == 1 else "No")
