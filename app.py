import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
try:
    model = joblib.load('xgb_pipeline.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Same feature engineering functions as in training
def map_state_to_region(s):
    north = ['ND','SD','MN','WI','MI','MT','IA','NE','IL','OH']
    south = ['TX','OK','AR','LA','MS','AL','GA','FL','TN','KY','NC','SC','VA','WV','PR']
    east  = ['PA','NY','NJ','MD','DE','CT','MA','VT','NH','RI','ME']
    west  = ['CA','WA','OR','NV','AZ','NM','UT','CO','ID','WY','AK','HI']
    if pd.isna(s): return 'Other'
    s = str(s).upper()
    if s in north: return 'North'
    if s in south: return 'South'
    if s in east:  return 'East'
    if s in west:  return 'West'
    return 'Other'

def simplify_income(inc):
    if inc in ['Not displayed', 'Not employed', 'Not available', None, np.nan]:
        return 'Unknown'
    if inc in ['Less than $25,000', '$25,000-49,999']:
        return 'Low'
    if inc in ['$50,000-74,999', '$75,000-99,999']:
        return 'Mid'
    if inc in ['$100,000+', '$100,000 or more']:
        return 'High'
    return 'Unknown'

def dti_category(dti):
    if pd.isna(dti): return 'Unknown DTI'
    if dti < 0.2: return 'Low DTI'
    if dti < 0.4: return 'Medium DTI'
    if dti < 0.6: return 'High DTI'
    return 'Very High DTI'

def credit_score_tier(score):
    if pd.isna(score): return 'Unknown Score'
    if score < 580: return 'Poor'
    if score < 670: return 'Fair'
    if score < 740: return 'Good'
    if score < 800: return 'Very Good'
    return 'Excellent'

def employment_duration_category(dur):
    if pd.isna(dur): return 'Unknown Duration'
    if dur < 12: return 'Short (<1yr)'
    if dur < 60: return 'Medium (1-5yrs)'
    return 'Long (>5yrs)'

def occupation_category(occupation):
    high_paying = [
        'Computer Programmer', 'Software Engineer', 'Engineer', 'Analyst',
        'Executive', 'Attorney', 'Doctor', 'Dentist', 'Pharmacist',
        'Scientist', 'Architect', 'Professor', 'Manager', 'Financial Advisor',
        'Physician', 'Registered Nurse', 'Teacher', 'Accountant', 'Sales - Commission'
    ]
    return 'Higher-Paying' if occupation in high_paying else 'Lower-Paying'

# Streamlit UI
st.title('Loan Default Prediction App')
st.write('Enter loan application details to predict the likelihood of default.')

b_state = st.selectbox('Borrower State', ['CA', 'NY', 'TX', 'FL', 'IL', 'OH', 'Other'])
credit_score_lower = st.slider('Credit Score Range Lower', 0, 850, 680)
credit_score_upper = st.slider('Credit Score Range Upper', 0, 850, 720)
income_range = st.selectbox('Income Range',
    ['Less than $25,000', '$25,000-49,999', '$50,000-74,999',
     '$75,000-99,999', '$100,000+', 'Not employed', 'Not displayed'], index=3)
employment_status = st.selectbox('Employment Status',
    ['Employed', 'Full-time', 'Self-employed', 'Retired', 'Other'])
dti_ratio = st.slider('Debt To Income Ratio', 0.0, 1.0, 0.3, 0.01)
emp_duration = st.slider('Employment Status Duration (months)', 0, 360, 60)
stated_monthly_income = st.number_input('Stated Monthly Income ($)', min_value=0.0, value=5000.0)
loan_original_amount = st.number_input('Loan Original Amount ($)', min_value=0.0, value=10000.0)
borrower_apr = st.number_input('Borrower APR', 0.0, 0.5, 0.15, 0.001)
borrower_rate = st.number_input('Borrower Rate', 0.0, 0.5, 0.14, 0.001)
occupation = st.selectbox('Occupation', [
    'Computer Programmer', 'Software Engineer', 'Engineer', 'Analyst', 'Executive',
    'Attorney', 'Doctor', 'Dentist', 'Pharmacist', 'Scientist', 'Architect',
    'Professor', 'Manager', 'Financial Advisor', 'Physician', 'Registered Nurse',
    'Teacher', 'Accountant', 'Sales - Commission', 'Other'
])

if st.button('Predict Loan Status'):
    input_data = {
        'BorrowerState': b_state,
        'CreditScoreRangeLower': credit_score_lower,
        'CreditScoreRangeUpper': credit_score_upper,
        'IncomeRange': income_range,
        'EmploymentStatus': employment_status,
        'DebtToIncomeRatio': dti_ratio,
        'EmploymentStatusDuration': emp_duration,
        'StatedMonthlyIncome': stated_monthly_income,
        'LoanOriginalAmount': loan_original_amount,
        'BorrowerAPR': borrower_apr,
        'BorrowerRate': borrower_rate,
        'Occupation': occupation
    }
    df_raw = pd.DataFrame([input_data])
    df_raw['Region'] = df_raw['BorrowerState'].apply(map_state_to_region)
    df_raw['CreditScoreAvg'] = (df_raw['CreditScoreRangeLower'] + df_raw['CreditScoreRangeUpper']) / 2
    df_raw['IncomeCat'] = df_raw['IncomeRange'].apply(simplify_income)
    df_raw['DebtToIncomeCategory'] = df_raw['DebtToIncomeRatio'].apply(dti_category)
    df_raw['CreditScoreTier'] = df_raw['CreditScoreAvg'].apply(credit_score_tier)
    df_raw['LoanToIncomeRatio'] = np.where(df_raw['StatedMonthlyIncome'] != 0,
                                           df_raw['LoanOriginalAmount'] / (df_raw['StatedMonthlyIncome'] * 12),
                                           np.nan)
    df_raw['LoanToIncomeRatio'].fillna(df_raw['LoanToIncomeRatio'].mean(), inplace=True)
    df_raw['EmploymentDurationCategory'] = df_raw['EmploymentStatusDuration'].apply(employment_duration_category)
    df_raw['OccupationCategory'] = df_raw['Occupation'].apply(occupation_category)

    feature_cols = [
        'StatedMonthlyIncome', 'LoanOriginalAmount', 'BorrowerAPR', 'BorrowerRate',
        'CreditScoreAvg', 'IncomeCat', 'Region', 'EmploymentStatus',
        'DebtToIncomeCategory', 'CreditScoreTier', 'LoanToIncomeRatio',
        'EmploymentDurationCategory', 'OccupationCategory'
    ]
    X_input = df_raw[feature_cols]

    pred = model.predict(X_input)
    proba = model.predict_proba(X_input)[0]

    st.subheader("Prediction Results")
    if pred[0] == 1:
        st.error("Predicted: **High Risk of Default**")
    else:
        st.success("Predicted: **Low Risk of Default**")
    st.write(f"Probability of Non-Default: {proba[0]:.2f} ({proba[0]*100:.1f}%)")
    st.write(f"Probability of Default:     {proba[1]:.2f} ({proba[1]*100:.1f}%)")
