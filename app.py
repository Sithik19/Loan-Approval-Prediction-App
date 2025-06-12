import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler from a combined pickle file
model, scaler = pk.load(open('loan_approval_model (7).pkl', 'rb'))

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'input'

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Input Page
def input_page():
    st.header('🏦 Loan Prediction App')

    no_of_dep = st.slider('Choose No of Dependents', 0, 5)
    grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
    self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
    Annual_Income = st.slider('Choose Annual Income (In INR)', 0, 10000000)
    Loan_Amount = st.slider('Choose Loan Amount (In INR)', 0, 10000000)
    Loan_Dur = st.slider('Choose Loan Duration (In Months)', 0, 20)
    Cibil = st.slider('Choose CIBIL Score', 0, 1000)
    Assets = st.slider('Choose Assets (In INR)', 0, 10000000)

    grad_s = 0 if grad == 'Graduated' else 1
    emp_s = 0 if self_emp == 'No' else 1

    if st.button("Predict"):
        # Validate inputs
        if Annual_Income <= 0 or Loan_Amount <= 0 or Loan_Dur <= 0 or Cibil <= 0 or Assets <= 0:
            st.warning("⚠️ Please fill in all fields with valid (non-zero) values.")
        else:
            # Prepare input data
            pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                     columns=['no_of_dependents', 'education', 'self_employed',
                                              'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
            # Scale and predict
            pred_data = scaler.transform(pred_data)
            prediction = model.predict(pred_data)

            # Store result and navigate to result page
            st.session_state.prediction = prediction[0]
            st.session_state.page = 'result'
            st.rerun()

# Result Page
def result_page():
    st.header("📋 Prediction Result")

    if st.session_state.prediction == 1:
        st.success("✅ Loan Is Approved")
    else:
        st.error("❌ Loan Is Rejected")

    if st.button("Back to Input"):
        st.session_state.page = 'input'
        st.rerun()

# Render correct page
if st.session_state.page == 'input':
    input_page()
elif st.session_state.page == 'result':
    result_page()
