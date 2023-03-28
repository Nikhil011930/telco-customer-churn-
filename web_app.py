import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the pre-trained XGBoost model
with open('D:/FastFinder course/Internship project/model.sav', 'rb') as f:
    model = pickle.load(f)


# Define the list of columns for user input
input_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']

# Define options for the categorical columns

def predict_churn(features):
    features = np.array(features).reshape(1,-1)
    
    prediction = model.predict(features)
    print('Input features:', features)
    print('Prediction:', prediction)
    if prediction == 1:
        return "Churn"
    else:
        return "No Churn"
    
def main():
    st.title('Churn Prediction')
   

    gender_options = ['Male', 'Female']
    yes_no_options = ['Yes', 'No']
    service_options = ['DSL', 'Fiber optic', 'No']
    device_options = yes_no_options + ['No internet service']
    contract_options = ['Month-to-month', 'One year', 'Two year']
    payment_options = ['Bank transfer (automatic)', 'Credit card (automatic)',
                    'Electronic check', 'Mailed check']
    

    # Collect user inputs
    gender = st.selectbox('Gender', gender_options)
    senior = st.selectbox('Senior Citizen', yes_no_options)
    partner = st.selectbox('Partner', yes_no_options)
    dependents = st.selectbox('Dependents', yes_no_options)
    tenure = st.number_input('Tenure (months):')
    phone_service = st.selectbox('Phone Service', yes_no_options)
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', service_options)
    online_security = st.selectbox('Online Security', device_options)
    online_backup = st.selectbox('Online Backup', device_options)
    device_protection = st.selectbox('Device Protection', device_options)
    tech_support = st.selectbox('Tech Support', device_options)
    streaming_tv = st.selectbox('Streaming TV', device_options)
    streaming_movies = st.selectbox('Streaming Movies', device_options)
    contract = st.selectbox('Contract', contract_options)
    paperless_billing = st.selectbox('Paperless Billing', yes_no_options)
    payment_method = st.selectbox('Payment Method', payment_options)
    monthly_charges = st.number_input('Monthly Charges ($):')
    total_charges = monthly_charges * tenure

    # Convert user inputs to a DataFrame
    features = pd.DataFrame({'gender': [gender],
                         'SeniorCitizen': [int(senior == 'Yes')],
                         'Partner': [partner],
                         'Dependents': [dependents],
                         'tenure': [tenure],
                         'PhoneService': [phone_service],
                         'MultipleLines': [multiple_lines],
                         'InternetService': [internet_service],
                         'OnlineSecurity': [online_security],
                         'OnlineBackup': [online_backup],
                         'DeviceProtection': [device_protection],
                         'TechSupport': [tech_support],
                         'StreamingTV': [streaming_tv],
                         'StreamingMovies': [streaming_movies],
                         'Contract': [contract],
                         'PaperlessBilling': [paperless_billing],
                         'PaymentMethod': [payment_method],
                         'MonthlyCharges': [monthly_charges],
                         'TotalCharges': [total_charges]})
    



    # Make predictions and display the results
    if st.button('Predict Churn Probability'):
            lab = LabelEncoder()
            cat_df_features = []
            # Get numerical features
            num_df_features = features.describe().columns.tolist()
            for i in list(features.columns):
                if i not in list(features.describe().columns):
                    cat_df_features.append(i)
            for i in cat_df_features:
                features[i] = lab.fit_transform(features[i])
                print(i,' : ',features[i].unique(),' = ',lab.inverse_transform(features[i].unique()))
            transformed_df = pd.concat([features[cat_df_features], features[num_df_features]], axis=1)
            transformed_df.drop(['PhoneService', 'gender', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'InternetService'], axis=1, inplace=True)
             
            result = predict_churn(transformed_df)
            st.write('The customer is likely to', result)
    
           

if __name__ == '__main__':
    main()

   