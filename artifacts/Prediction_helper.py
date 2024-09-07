from joblib import load
import pandas as pd

#link = r'C:\Users\91868\OneDrive\pgp\Python\3. Machine learning\7. Projects\Health care premium\Premium_Prediction_App\\'
model_rest = load('artifacts/model_rest.joblib')
model_young = load('artifacts/model_young.joblib')

scaler_rest = load('artifacts/scaler_rest.joblib')
scaler_young = load('artifacts/scaler_young.joblib')

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def preprocess_input(input_dict):
    expected_column = ['Age','Number_Of_Dependants','Income_Lakhs',	'Insurance_Plan','Genetical_Risk','normalized_risk_score','Gender_Male',
                       'Region_Northwest','Region_Southeast','Region_Southwest','Marital_status_Unmarried','BMI_Category_Obesity',
                       'BMI_Category_Overweight','BMI_Category_Underweight','Smoking_Status_Occasional',
                       'Smoking_Status_Regular','Employment_Status_Salaried','Employment_Status_Self-Employed']
    
    Insurance_Plan_encoding = {'Bronze': 1,'Silver': 2,'Gold': 3}
    
    df = pd.DataFrame(0, columns=expected_column, index=[0])    

    # Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['Gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['Region_Northwest'] = 1
            elif value == 'Southeast':
                df['Region_Southeast'] = 1
            elif value == 'Southwest':
                df['Region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['Marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['BMI_Category_Obesity'] = 1
            elif value == 'Overweight':
                df['BMI_Category_Overweight'] = 1
            elif value == 'Underweight':
                df['BMI_Category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['Smoking_Status_Occasional'] = 1
            elif value == 'Regular':
                df['Smoking_Status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['Employment_Status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['Employment_Status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  
            df['Insurance_Plan'] = Insurance_Plan_encoding.get(value, 1)
        elif key == 'Age':  
            df['Age'] = value  # Corrected to 'Age' with capital 'A'
        elif key == 'Number of Dependants':  
            df['Number_Of_Dependants'] = value
        elif key == 'Income in Lakhs':  
            df['Income_Lakhs'] = value
        elif key == "Genetical Risk":
            df['Genetical_Risk'] = value

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'Medical History'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['Income_Level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('Income_Level', axis='columns', inplace=True)

    return df   

 

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])