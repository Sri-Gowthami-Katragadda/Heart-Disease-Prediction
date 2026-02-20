# import streamlit as st
# import pandas as pd
# import joblib

# # Load saved model, scaler, and expected columns
# model = joblib.load("Logistic_heart_pred.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")

# st.title("Heart Stroke Prediction by Gowthiii")
# st.markdown("Provide the following details to check your heart stroke risk:")

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# time= st.slider("Time", 1, 500, 30)
# anaemia = st.selectbox("Anaemia", ["Yes", "No"])
# diabetes = st.selectbox("Diabetes", ["Yes", "No"])
# smoking= st.selectbox("Smoking", ["Yes","no"])
# sex= st.selectbox("Sex", ["M","F"])
# creatinine_phosphokinase= st.number_input("Creatinine Phosphokinase", 50, 10000,1000 )
# ejection_fraction = st.number_input("Ejection Fraction", 0,200 , 50)
# high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
# platelets= st.number_input("Platelets",0,1000000,20000)
# serum_creatinine= st.number_input("Serum Creatinine", 0, 10, 5)
# serum_sodium = st.number_input("Serum Sodium", 40,200, 50)


# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'Anaemia': anaemia,
#         'Diabetes': diabetes,
#         'Smoking': smoking,
#         'Creatinine Phosphokinase': creatinine_phosphokinase,
#         'Ejection Fraction': ejection_fraction,
#         'Time':time,
#         'Sex_' + sex: 1,
#         'High Blood Pressure_' :high_blood_pressure,
#         "Platelets":platelets,
#         "Serum Creatinine":serum_creatinine,
#         "Serum Sodium":serum_sodium
#         # 'RestingECG_' + resting_ecg: 1,
#         # 'ExerciseAngina_' + exercise_angina: 1,
#         # 'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scaler.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")


import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Saved Files
# -----------------------------
model = joblib.load("Logistic_heart_pred.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("❤️ Heart Failure Prediction App")
st.markdown("Provide patient details to check heart failure risk.")

st.markdown("### Clinical Inputs")

# -----------------------------
# User Inputs (MATCH TRAINING FORMAT)
# -----------------------------

age = st.slider("Age", 40, 95, 60)

anaemia = st.selectbox("Anaemia", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
sex = st.selectbox("Sex", ["Female", "Male"])
high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])

creatinine_phosphokinase = st.number_input(
    "Creatinine Phosphokinase", 23, 8000, 250
)

ejection_fraction = st.slider("Ejection Fraction (%)", 14, 80, 38)

platelets = st.number_input("Platelets", 25000, 850000, 250000)

serum_creatinine = st.number_input("Serum Creatinine", 0.5, 9.4, 1.0)

serum_sodium = st.slider("Serum Sodium", 113, 148, 137)

time = st.slider("Follow-up Period (Days)", 4, 285, 120)

# -----------------------------
# Convert Yes/No → 1/0 (IMPORTANT)
# -----------------------------
anaemia = 1 if anaemia == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
smoking = 1 if smoking == "Yes" else 0
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
sex = 1 if sex == "Male" else 0   # Dataset uses 1 = Male, 0 = Female

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # IMPORTANT: Column names must EXACTLY match training dataset
    input_data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure same column order as training
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # -----------------------------
    # Show Result
    # -----------------------------
    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Failure\n\nRisk Score: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Failure\n\nRisk Score: {probability:.2f}")
