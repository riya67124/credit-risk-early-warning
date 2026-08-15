import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Early-Warning System")
st.write("Enter applicant details to check credit risk")

model = joblib.load('credit_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def predict_new_applicant(applicant_dict):
    new_df = pd.DataFrame([applicant_dict])
    categorical_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status',
                         'employment', 'personal_status', 'other_debtors', 'property',
                         'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']
    new_encoded = pd.get_dummies(new_df, columns=categorical_cols)
    new_encoded = new_encoded.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(new_encoded)[0]
    probability = model.predict_proba(new_encoded)[0]
    risk_label = 'good' if prediction == 1 else 'bad'
    return risk_label, round(probability[1] * 100, 2)

# --- Form ---
checking_status = st.selectbox("Checking Account Status", ['<0 DM', '0-200 DM', '>=200 DM', 'no account'])
duration = st.slider("Loan Duration (months)", 4, 72, 24)
credit_history = st.selectbox("Credit History", ['no credits/all paid', 'all paid this bank', 'existing paid duly', 'delay in past', 'critical account'])
purpose = st.selectbox("Loan Purpose", ['new car', 'used car', 'furniture', 'radio/TV', 'appliances', 'repairs', 'education', 'retraining', 'business', 'other'])
credit_amount = st.number_input("Credit Amount (DM)", 250, 20000, 3000)
savings_status = st.selectbox("Savings Account", ['<100 DM', '100-500 DM', '500-1000 DM', '>=1000 DM', 'unknown/none'])
employment = st.selectbox("Employment Duration", ['unemployed', '<1 year', '1-4 years', '4-7 years', '>=7 years'])
installment_rate = st.slider("Installment Rate (% of income)", 1, 4, 2)
personal_status = st.selectbox("Personal Status", ['male-divorced', 'female-div/married', 'male-single', 'male-married', 'female-single'])
other_debtors = st.selectbox("Other Debtors/Guarantors", ['none', 'co-applicant', 'guarantor'])
residence_since = st.slider("Years at Current Residence", 1, 4, 2)
property_ = st.selectbox("Property", ['real estate', 'savings/insurance', 'car/other', 'unknown/none'])
age = st.slider("Age", 18, 75, 30)
other_installment_plans = st.selectbox("Other Installment Plans", ['bank', 'stores', 'none'])
housing = st.selectbox("Housing", ['rent', 'own', 'free'])
existing_credits = st.slider("Existing Credits at This Bank", 1, 4, 1)
job = st.selectbox("Job", ['unemployed/unskilled', 'unskilled-resident', 'skilled', 'management/highly qualified'])
dependents = st.slider("Number of Dependents", 1, 2, 1)
telephone = st.selectbox("Telephone", ['none', 'yes'])
foreign_worker = st.selectbox("Foreign Worker", ['yes', 'no'])

if st.button("Check Credit Risk"):
    applicant = {
        'checking_status': checking_status, 'duration': duration, 'credit_history': credit_history,
        'purpose': purpose, 'credit_amount': credit_amount, 'savings_status': savings_status,
        'employment': employment, 'installment_rate': installment_rate, 'personal_status': personal_status,
        'other_debtors': other_debtors, 'residence_since': residence_since, 'property': property_,
        'age': age, 'other_installment_plans': other_installment_plans, 'housing': housing,
        'existing_credits': existing_credits, 'job': job, 'dependents': dependents,
        'telephone': telephone, 'foreign_worker': foreign_worker
    }
    label, prob = predict_new_applicant(applicant)
    if label == 'bad':
        st.error(f"⚠️ High Risk — {prob}% probability of default")
    else:
        st.success(f"✅ Low Risk — {prob}% probability of default")
