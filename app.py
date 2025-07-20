import streamlit as st
import numpy as np
import joblib

import gdown
import os
import joblib

# Download model.pkl
if not os.path.exists("model.pkl"):
    gdown.download("https://drive.google.com/uc?id=1Gy9TXenOXMcMhzH_zlR-sLTCvjnQhjQq", "model.pkl", quiet=False)

# Download scaler.pkl
if not os.path.exists("scaler.pkl"):
    gdown.download("https://drive.google.com/uc?id=1uk_bNe1YQqXfbVzeKFbETWgkWpgNN8j9", "scaler.pkl", quiet=False)

# Download encoders.pkl
if not os.path.exists("encoders.pkl"):
    gdown.download("https://drive.google.com/uc?id=1nycPX35a4OeovsOq5Esz7n7CwYpkG5nU", "encoders.pkl", quiet=False)

# Load the files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")



if 'show_form' not in st.session_state:
    st.session_state.show_form = True

st.set_page_config(page_title="Income Category & Financial Advisor 💰", page_icon="💸")

st.title("Income Prediction & Personalized Financial Advice 💡💰")
st.write("Get an instant income prediction and **personalized financial recommendations** based on your profile. 🧠📈")
# Model Info Section
st.markdown("### 🔍 About This Predictor")
st.markdown("""
This tool uses a **Random Forest Classifier** trained on US Census data to predict income levels.

**Model Performance:**
- ✅ Accuracy: **85.43%**
- 📈 ROC AUC: **90.55%**

Fill in your details below to get personalized predictions and financial advice. 🧾💸
""")
if st.session_state.show_form:

    st.header("📝 Enter Your Details")


# User Inputs
    age = st.number_input("Enter your Age:", min_value=18, max_value=100, step=1)

    workclass = st.selectbox("Work Class 🏢", encoders['workclass'].classes_)
    education_map = {
        1: "Preschool",
        2: "1st-4th",
        3: "5th-6th",
        4: "7th-8th",
        5: "9th",
        6: "10th",
        7: "11th",
        8: "12th",
        9: "High School Grad",
        10: "Some College",
        11: "Associate (Vocational)",
        12: "Associate (Academic)",
        13: "Bachelors",
        14: "Masters",
        15: "Doctorate",
        16: "Professional School"
}
    
    education_label = st.selectbox("Education 🎓", list(education_map.values()))
    education_num = [k for k, v in education_map.items() if v == education_label][0]
    
    marital_status = st.selectbox("Marital Status 💍", encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation 💼", encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship 👨‍👩‍👧‍👦", encoders['relationship'].classes_)
    race = st.selectbox("Race 🌍", encoders['race'].classes_)
    gender = st.selectbox("Gender ⚧️", encoders['gender'].classes_)
    hours_per_week = st.slider("Hours Worked per Week ⏱️", 1, 99, 40)
    native_country = st.selectbox("Country 🌐", encoders['native-country'].classes_)
    capital_gain = st.number_input("Capital Gain 📈", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss 📉", 0, 100000, 0)

    # Prepare Input
    input_data = {
        'age': age,
        'workclass': workclass,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    
# Encode categorical columns
    for col in encoders:
        input_data[col] = encoders[col].transform([input_data[col]])[0]

# Convert to feature array
    features = np.array([[input_data['age'], input_data['workclass'], input_data['education-num'],
                        input_data['marital-status'], input_data['occupation'], input_data['relationship'],
                        input_data['race'], input_data['gender'], input_data['capital-gain'], input_data['capital-loss'],
                        input_data['hours-per-week'], input_data['native-country']]])

# Scale features
    features_scaled = scaler.transform(features)
    
    # Predict Button (Inside this block)
    if st.button("Predict Income Category 🚀"):
        st.session_state.show_form = False  # Hide form after click
        st.session_state.predicted = True

        # Store features in session state to use later
        st.session_state.features_scaled = scaler.transform(features)
if 'predicted' in st.session_state and st.session_state.predicted:

    pred = model.predict(st.session_state.features_scaled)[0]

    if pred == 1:
        st.balloons()
        st.success("🎉 **Predicted Income: >50K**")
        st.markdown("""
        ### 🤑 **Recommended Financial Actions**
        - 💎 **Premium Health & Life Insurance (10-25L Cover)**
        - 📈 **Invest in Mutual Funds, Stocks, Real Estate**
        - 🧾 **Plan for Tax Savings (ELSS, NPS)**
        - 🏦 **Consider Wealth Management Services**
        - 📜 **Estate Planning: Create a Will**
        """)
    else:
        st.warning("🔔 **Predicted Income: ≤50K**")
        st.markdown("""
        ### 🛡️ **Recommended Financial Actions**
        - 🏥 **Join Government Health Schemes (Ayushman Bharat)**
        - 📑 **Get PMJJBY Insurance (₹330/year)**
        - 🎓 **Upskilling via Skill India, PMKVY**
        - 🏦 **Open a Jan Dhan Bank Account**
        - 🛍️ **Apply for Subsidies & Support Schemes**
        """)

 
st.markdown("---")
st.caption("Made with ❤️ by Bhanu's AI Income Profiler 🚀")
