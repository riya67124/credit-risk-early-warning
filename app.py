import streamlit as st
import pandas as pd
import joblib
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Risk System", layout="wide")

# --- Custom styling: bigger, centered tabs + overall polish ---
st.markdown("""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 40px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 28px;
        font-size: 17px;
        font-weight: 600;
        background-color: transparent;
        border-bottom: 3px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #2C6E91;
        border-bottom: 3px solid #2C6E91;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2C6E91;
        background-color: transparent;
    }
    div.block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    </style>
""", unsafe_allow_html=True)

PALETTE = "Blues_r"
GOOD_COLOR = "#4C9F70"
BAD_COLOR = "#D65F5F"

model = joblib.load('credit_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def get_connection():
    return psycopg2.connect(st.secrets["connections"]["postgres"]["url"], options="-c search_path=public")

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
    return risk_label, round(float(probability[1]) * 100, 2)

def save_submission(applicant_dict, risk_label, probability):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.applicant_submissions 
        (checking_status, duration, credit_history, purpose, credit_amount, savings_status,
         employment, installment_rate, personal_status, other_debtors, residence_since, property,
         age, other_installment_plans, housing, existing_credits, job, dependents, telephone,
         foreign_worker, predicted_risk, bad_risk_probability)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        applicant_dict['checking_status'], applicant_dict['duration'], applicant_dict['credit_history'],
        applicant_dict['purpose'], applicant_dict['credit_amount'], applicant_dict['savings_status'],
        applicant_dict['employment'], applicant_dict['installment_rate'], applicant_dict['personal_status'],
        applicant_dict['other_debtors'], applicant_dict['residence_since'], applicant_dict['property'],
        applicant_dict['age'], applicant_dict['other_installment_plans'], applicant_dict['housing'],
        applicant_dict['existing_credits'], applicant_dict['job'], applicant_dict['dependents'],
        applicant_dict['telephone'], applicant_dict['foreign_worker'], risk_label, float(probability)
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_dashboard_data():
    conn = get_connection()
    total = pd.read_sql("SELECT COUNT(*) as total FROM public.applicant_submissions;", conn)
    by_risk = pd.read_sql("SELECT predicted_risk, COUNT(*) as count FROM public.applicant_submissions GROUP BY predicted_risk;", conn)
    by_purpose = pd.read_sql("""
        SELECT purpose, COUNT(*) as total,
               ROUND(100.0 * SUM(CASE WHEN predicted_risk='bad' THEN 1 ELSE 0 END) / COUNT(*), 1) as bad_pct
        FROM public.applicant_submissions GROUP BY purpose ORDER BY bad_pct DESC;
    """, conn)
    by_employment = pd.read_sql("""
        SELECT employment, COUNT(*) as total,
               ROUND(100.0 * SUM(CASE WHEN predicted_risk='bad' THEN 1 ELSE 0 END) / COUNT(*), 1) as bad_pct
        FROM public.applicant_submissions GROUP BY employment ORDER BY bad_pct DESC;
    """, conn)
    amounts = pd.read_sql("SELECT predicted_risk, credit_amount FROM public.applicant_submissions;", conn)
    recent = pd.read_sql("""
        SELECT submitted_at, purpose, credit_amount, predicted_risk, bad_risk_probability 
        FROM public.applicant_submissions ORDER BY submitted_at DESC LIMIT 10;
    """, conn)
    conn.close()
    return total, by_risk, by_purpose, by_employment, amounts, recent

# --- Top navbar-style tabs (styled bigger + centered via CSS above) ---
tab1, tab2 = st.tabs(["🔍  Predict", "📊  Dashboard"])

with tab1:
    st.title("Credit Risk Early-Warning System")
    st.write("Enter applicant details to check credit risk")

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
        save_submission(applicant, label, prob)
        if label == 'bad':
            st.error(f"⚠️ High Risk — {prob}% probability of default")
        else:
            st.success(f"✅ Low Risk — {prob}% probability of default")

with tab2:
    st.title("Live Usage Dashboard")
    st.write("Real-time stats from every prediction run through this app")

    total, by_risk, by_purpose, by_employment, amounts, recent = get_dashboard_data()

    if total['total'][0] == 0:
        st.info("No submissions yet — try the Predict tab first!")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Submissions", int(total['total'][0]))
        bad_count = by_risk[by_risk['predicted_risk']=='bad']['count'].sum() if 'bad' in by_risk['predicted_risk'].values else 0
        col2.metric("Flagged High Risk", int(bad_count))
        col3.metric("Final Model", "Logistic Regression")

        st.markdown("####")  # small spacer

        # --- 4 charts, one row ---
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.caption("Good vs Bad Split")
            fig1, ax1 = plt.subplots(figsize=(3.5, 3))
            colors = {'good': GOOD_COLOR, 'bad': BAD_COLOR}
            ax1.pie(by_risk['count'], labels=by_risk['predicted_risk'], autopct='%1.0f%%',
                    colors=[colors.get(r, '#999') for r in by_risk['predicted_risk']],
                    textprops={'fontsize': 8})
            st.pyplot(fig1, use_container_width=True)

        with c2:
            st.caption("Risk % by Purpose")
            if len(by_purpose) > 0:
                fig2, ax2 = plt.subplots(figsize=(3.5, 3))
                sns.barplot(x='bad_pct', y='purpose', data=by_purpose.head(6), ax=ax2,
                            hue='purpose', palette=PALETTE, legend=False)
                ax2.set_xlabel("Bad %", fontsize=8)
                ax2.set_ylabel("")
                ax2.tick_params(labelsize=7)
                st.pyplot(fig2, use_container_width=True)

        with c3:
            st.caption("Risk % by Employment")
            if len(by_employment) > 0:
                fig3, ax3 = plt.subplots(figsize=(3.5, 3))
                sns.barplot(x='bad_pct', y='employment', data=by_employment, ax=ax3,
                            hue='employment', palette=PALETTE, legend=False)
                ax3.set_xlabel("Bad %", fontsize=8)
                ax3.set_ylabel("")
                ax3.tick_params(labelsize=7)
                st.pyplot(fig3, use_container_width=True)

        with c4:
            st.caption("Credit Amount by Risk")
            if len(amounts) > 0:
                fig4, ax4 = plt.subplots(figsize=(3.5, 3))
                sns.boxplot(x='predicted_risk', y='credit_amount', data=amounts, ax=ax4,
                            hue='predicted_risk', palette={'good': GOOD_COLOR, 'bad': BAD_COLOR}, legend=False)
                ax4.set_xlabel("")
                ax4.set_ylabel("Amount", fontsize=8)
                ax4.tick_params(labelsize=7)
                st.pyplot(fig4, use_container_width=True)

        st.markdown("####")
        st.subheader("Most Recent Submissions")
        st.dataframe(recent, use_container_width=True)
