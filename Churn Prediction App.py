# TELECOM CUSTOMER CHURN PREDICTION APP
## Six key steps in building a web app:
# 1. Import required libraries
# 2. Load your trained machine learning model
# 3. Write a function that will make the prediction
# 4. Build your Streamlit interface
# 5. Set up prediction button
# 6. Run the web app

import os
import pickle          # Helps load the trained model / scaler / column list
import numpy as np     # Convert user data to numpy array
import pandas as pd    # Build the single-row feature frame in the right col
import streamlit as st


# STEP 1 & 2: LOAD TRAINED MODEL
load_model = pickle.load(open('model.sav', 'rb'))

# Exact feature columns/order the trained pipeline was fit on (x_train.columns
# from the notebook). input_data is built as a dict below so these names line
# up correctly with the model instead of relying on plain list position.
FEATURE_COLUMNS = [
    "age", "gender", "plan_type", "monthly_data_usage_gb", "monthly_voice_minutes",
    "monthly_sms_count", "monthly_spend", "tenure_months", "dropped_call_rate",
    "num_customer_care_calls", "num_complaints", "has_promo_subscription",
    "roaming_user", "last_recharge_days_ago",
    "state_Abuja", "state_Anambra", "state_Delta", "state_Enugu", "state_Kaduna",
    "state_Kano", "state_Lagos", "state_Ogun", "state_Oyo", "state_Rivers",
    "network_provider_BlueWave Mobile", "network_provider_ConnectPlus",
    "network_provider_NovaTel", "network_provider_Zenith Networks",
    "data_plan_Basic", "data_plan_Monthly", "data_plan_Unlimited", "data_plan_Weekly",
    "network_type_2G", "network_type_3G", "network_type_4G", "network_type_5G",
]


# STEP 3: PREDICTION FUNCTION
def churn_prediction(input_data):

    # Convert input into a DataFrame (input_data is a dict, so keys become
    # column names in the right order for the model)
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Make prediction
    prediction = load_model.predict(input_df)

    # Get probability
    probability = load_model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        return "High Churn Risk", probability
    else:
        return "Low Churn Risk", probability

# STEP 4: STREAMLIT INTERFACE

def main():
    # Page configuration
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📱", layout="wide")

    # App heading
    st.title("📱 Telecom Customer Churn Predictor")
    st.write("Enter the customer's details below to predict their likelihood of churning.")


# CUSTOMER INFORMATION
# customer_id,age,gender,plan_type,monthly_data_usage_gb,monthly_voice_minutes,monthly_sms_count,monthly_spend,tenure_months,
# ,last_recharge_days_ago,churn,
# network_provider_BlueWave Mobile,network_provider_ConnectPlus,network_provider_NovaTel,network_provider_Zenith 
# Networks,data_plan_Basic,data_plan_Monthly,data_plan_Unlimited,data_plan_Weekly,network_type_2G,network_type_3G,network_type_4G,
#network_type_5G

    # Using 3 columns for inputs
    col1, col2, col3 = st.columns(3)


    # COLUMN 1
    with col1:
        st.subheader("Customer Information")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        state = st.selectbox("State", options=["Enugu", "Kano", "Delta", "Lagos", "Abuja", "Kaduna", "Oyo", "Anambra", "Ogun", "Rivers"])
        tenure_months = st.number_input("Tenure (Months)", min_value=0, value=24, step=1)


    # COLUMN 2
    with col2:
        st.subheader("Service Information")
        network_provider = st.selectbox("Network Provider", options=["ConnectPlus", "NovaTel", "Zenith Networks", "BlueWave Mobile"])
        network_type = st.selectbox("Network Type", options=["2G", "3G", "4G", "5G"])
        plan_type = st.selectbox("Plan Type", options=["Prepaid", "Postpaid"])
        data_plan = st.selectbox("Data Plan", options=["Weekly", "Monthly", "Basic", "Unlimited"])
        roaming_user = st.selectbox("Roaming User", options=["No", "Yes"])


    # COLUMN 3
    with col3:
        st.subheader("Usage & Spending")
        monthly_data_usage_gb = st.number_input("Monthly Data Usage (GB)", min_value=0.0, value=10.0, step=1.0)
        monthly_voice_minutes = st.number_input("Monthly Voice Minutes", min_value=0.0, value=500.0, step=10.0)
        monthly_sms_count = st.number_input("Monthly SMS Count", min_value=0.0, value=100.0, step=5.0)
        monthly_spend = st.number_input("Monthly Spend (₦)", min_value=0.0, value=5000.0, step=100.0)


    # CUSTOMER BEHAVIOUR

    st.markdown("---")
    st.subheader("Customer Behaviour")

    col4, col5, col6 = st.columns(3)
    with col4:
        dropped_call_rate = st.number_input("Dropped Call Rate", min_value=0.0, max_value=20.0, value=0.05, step=0.01)
        num_customer_care_calls = st.number_input("Customer Care Calls", min_value=0, value=2, step=1)

    with col5:
        num_complaints = st.number_input("Number of Complaints", min_value=0, value=0, step=1)
        last_recharge_days_ago = st.number_input("Days Since Last Recharge", min_value=0, value=7, step=1)

    with col6:
        has_promo_subscription = st.selectbox("Promo Subscription", options=["No", "Yes"])


    # STEP 5: PREDICTION BUTTON

    st.markdown("---")

    if st.button("Predict Customer Churn", use_container_width=True):
        if load_model is None:
            st.error("Model is not loaded. Please make sure model.sav exists.")
            return
        try:
            # Feature Encoding (matching the columns the pipeline was trained on)
            # Convert Yes / No fields
            promo_value = 1 if has_promo_subscription == "Yes" else 0
            roaming_value = 1 if roaming_user == "Yes" else 0
            gender_val = 1 if gender == "Male" else 0
            plan_type_val = 1 if plan_type == "Postpaid" else 0

            # One-hot encoding
            network_provider_BlueWave_Mobile = 1 if network_provider == 'BlueWave Mobile' else 0
            network_provider_ConnectPlus = 1 if network_provider == 'ConnectPlus' else 0
            network_provider_NovaTel = 1 if network_provider == 'NovaTel' else 0
            network_provider_Zenith_Networks = 1 if network_provider == 'Zenith Networks' else 0

            data_plan_Basic = 1 if data_plan == 'Basic' else 0
            data_plan_Monthly = 1 if data_plan == 'Monthly' else 0
            data_plan_Unlimited = 1 if data_plan == 'Unlimited' else 0
            data_plan_Weekly = 1 if data_plan == 'Weekly' else 0

            network_type_2G = 1 if network_type == '2G' else 0
            network_type_3G = 1 if network_type == '3G' else 0
            network_type_4G = 1 if network_type == '4G' else 0
            network_type_5G = 1 if network_type == '5G' else 0

            state_Abuja = 1 if state == 'Abuja' else 0
            state_Anambra = 1 if state == 'Anambra' else 0
            state_Delta = 1 if state == 'Delta' else 0
            state_Enugu = 1 if state == 'Enugu' else 0
            state_Kaduna = 1 if state == 'Kaduna' else 0
            state_Kano = 1 if state == 'Kano' else 0
            state_Lagos = 1 if state == 'Lagos' else 0
            state_Ogun = 1 if state == 'Ogun' else 0
            state_Oyo = 1 if state == 'Oyo' else 0
            state_Rivers = 1 if state == 'Rivers' else 0

            # Create customer information (dict, not a plain list, so the
            # DataFrame column names match what the model was trained on)
            input_data = {
                "age": age,
                "gender": gender_val,
                "plan_type": plan_type_val,
                "monthly_data_usage_gb": monthly_data_usage_gb,
                "monthly_voice_minutes": monthly_voice_minutes,
                "monthly_sms_count": monthly_sms_count,
                "monthly_spend": monthly_spend,
                "tenure_months": tenure_months,
                "dropped_call_rate": dropped_call_rate,
                "num_customer_care_calls": num_customer_care_calls,
                "num_complaints": num_complaints,
                "has_promo_subscription": promo_value,
                "roaming_user": roaming_value,
                "last_recharge_days_ago": last_recharge_days_ago,
                "state_Abuja": state_Abuja,
                "state_Anambra": state_Anambra,
                "state_Delta": state_Delta,
                "state_Enugu": state_Enugu,
                "state_Kaduna": state_Kaduna,
                "state_Kano": state_Kano,
                "state_Lagos": state_Lagos,
                "state_Ogun": state_Ogun,
                "state_Oyo": state_Oyo,
                "state_Rivers": state_Rivers,
                "network_provider_BlueWave Mobile": network_provider_BlueWave_Mobile,
                "network_provider_ConnectPlus": network_provider_ConnectPlus,
                "network_provider_NovaTel": network_provider_NovaTel,
                "network_provider_Zenith Networks": network_provider_Zenith_Networks,
                "data_plan_Basic": data_plan_Basic,
                "data_plan_Monthly": data_plan_Monthly,
                "data_plan_Unlimited": data_plan_Unlimited,
                "data_plan_Weekly": data_plan_Weekly,
                "network_type_2G": network_type_2G,
                "network_type_3G": network_type_3G,
                "network_type_4G": network_type_4G,
                "network_type_5G": network_type_5G,
            }

            # Make prediction
            result, probability = churn_prediction(input_data)


            # DISPLAY RESULT
            st.markdown("---")

            if result == "High Churn Risk":
                st.error("⚠️ HIGH CHURN RISK")
                st.write("This customer is likely to churn.")
            else:
                st.success("✅ LOW CHURN RISK")
                st.write("This customer is likely to remain.")

            # Display probability
            st.metric("Churn Probability", f"{probability:.1%}")

            st.progress(float(probability))


        except ValueError as ve:
            st.error(f'ValueError (Check feature shape): {ve}')
        except Exception as e:
            st.error(f"Prediction Error: {e}")


# STEP 6: RUN APP

if __name__ == "__main__":
    main()
