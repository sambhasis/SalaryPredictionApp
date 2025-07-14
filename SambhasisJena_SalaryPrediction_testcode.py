import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="Salary Predictor App", layout="wide")

# Inject CSS
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    #background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
    background-color:black;
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
}
.block-container {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 2rem;
    border-radius: 10px;
    color: white;
}
.stButton button, .stDownloadButton button {
    background-color: #ffffff;
    color: black;
    font-weight: bold;
    padding: 0.5rem 1.5rem;
    border-radius: 8px;
    border: none;
}
/* Home text */
.big-font {
    font-size: 20px;
    color: #F8F9FA;
}

/* Sidebar buttons uniform style */
button[kind="secondary"] {
    width: 100% !important;
    max-width: 220px !important;
    height: 48px !important;
    background-color: white !important;
    color: black !important;
    font-size: 16px !important;
    font-weight: bold !important;
    margin: 8px auto;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s ease;
}
button[kind="secondary"]:hover {
    background-color: #f0f0f0 !important;
}
button[kind="secondary"] > div {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 16px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Load model and encoders
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Initialize session page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Navigation
with st.sidebar:
    st.markdown("### ☰ Menu")
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.button("📝 Manual Entry"):
        st.session_state.page = "Manual Entry"
    if st.button("📂 Bulk Upload"):
        st.session_state.page = "Bulk Upload"
    if st.button("💬 Feedback"):
        st.session_state.page = "Feedback"

# ---------------------------- Pages ---------------------------- #

# Home Page
if st.session_state.page == "Home":
    st.title("💼 Welcome to Salary Predictor App")
    st.markdown("""
    <div class='big-font'>
    This application allows users to predict employee salaries based on various factors such as education level, experience, job role, and more.

    <strong>You can choose:</strong><br><br>
    <ul>
        <li><strong>Manual Entry</strong> – to input one employee’s details and get a prediction</li>
        <li><strong>Bulk Upload</strong> – to upload a CSV file and predict for many employees at once</li>
    </ul>

    <br><br>
    <strong>Developed by:</strong> Sambhasis Jena<br>
    <strong>Contact:</strong> sambhasis.jena245@gmail.com<br><br>
    © 2025 Sambhasis Jena. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# Manual Entry Page
elif st.session_state.page == "Manual Entry":
    st.title("🔹 Manual Entry")
    st.markdown("### Enter employee details:")

    col1, col2, col3 = st.columns(3)
    with col1:
        education_level = st.selectbox("Education Level", ["High School", "Bachelors", "PhD","Masters"])
        years_experience = st.slider("Years of Experience", 0, 50, 3)
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    with col2:
        job_title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "Analyst", "Manager"])
        age = st.slider("Age", 18, 80, 30)
        industry = st.selectbox("Industry", ["IT", "Education", "Healthcare", "Finance"])
    with col3:
        certifications = st.number_input("Number of Certifications", 0, 10, 0)
        working_hours = st.slider("Working Hours per Week", 0, 100, 40)
        location = st.selectbox("Location", ["New York","Bangalore","London","San Francisco"])

    input_dict = {
        "education_level": education_level,
        "years_experience": years_experience,
        "job_title": job_title,
        "industry": industry,
        "location": location,
        "company_size": company_size,
        "certifications": certifications,
        "age": age,
        "working_hours": working_hours
    }

    input_df = pd.DataFrame([input_dict])

    for col in input_df.select_dtypes(include='object').columns:
        le = label_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except:
                st.error(f"Invalid value for '{col}'")
                st.stop()

    if st.button("🔍 Predict Salary"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: **${prediction:,.2f}**")

# Bulk Upload Page
elif st.session_state.page == "Bulk Upload":
    st.title("📁 Bulk Upload")
    st.markdown("### Upload a CSV file to predict salaries for multiple entries.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    sample_df = pd.DataFrame({
        "education_level": ["Bachelors"],
        "years_experience": [5],
        "job_title": ["Data Scientist"],
        "industry": ["IT"],
        "location": ["New York"],
        "company_size": ["Medium"],
        "certifications": [2],
        "age": [30],
        "working_hours": [40]
    })
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Sample CSV", sample_csv, file_name="sample_salary_input.csv", mime="text/csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🧾 Uploaded CSV Preview:")
        st.dataframe(df.head())

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        for col in df.select_dtypes(include='object').columns:
            le = label_encoders.get(col)
            if le:
                try:
                    df[col] = le.transform(df[col])
                except:
                    st.error(f"Value error in column: {col}. Please correct input file.")
                    st.stop()

        if st.button("🔍 Predict for All Rows"):
            predictions = model.predict(df)
            df["Predicted_Salary"] = predictions
            st.success("Predictions completed!")
            st.dataframe(df)
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", csv_out, "salary_predictions.csv", "text/csv")

# Feedback Page
elif st.session_state.page == "Feedback":
    st.title("💬 Feedback / Suggestions")
    st.markdown("""
    If you have any suggestions, feedback, or encounter issues, feel free to contact:

    📧 **sambhasis.jena245@gmail.com**

    We appreciate your input to help improve the app!
    """)
