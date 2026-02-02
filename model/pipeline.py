from utils import pd, prepared_data

# Put the CSV file folder than the python files or change the path bellow
raw_data = pd.read_csv('stroke_data.csv', index_col=0)

# Keep the needed features and clean the data by removing the NaN and noisy values
df_processed = raw_data.reset_index(drop=True).drop(columns=['ever_married', 'Residence_type']).loc[lambda df: df['gender'] != 'Other']
df_processed['bmi'] = df_processed['bmi'].fillna(df_processed.groupby('gender')['bmi'].transform('median'))
df_processed['smoking_status'] = df_processed['smoking_status'].replace({ 'Unknown': 'never smoked', 'formerly smoked': 'smokes' })
df_processed = pd.get_dummies(df_processed, columns=['gender', 'work_type', 'smoking_status'], drop_first=True)

# Computation of new complexe features
df_processed['age_glucose'] = df_processed['age'] * df_processed['avg_glucose_level']
df_processed['age_hypertension'] = df_processed['age'] * df_processed['hypertension']
df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
df_processed['age_heart'] = df_processed['age'] * df_processed['heart_disease']
df_processed['risk_factors_count'] = df_processed['hypertension'] + df_processed['heart_disease'] + (df_processed['avg_glucose_level'] > 150).astype(int)
df_processed['risk_factors_increaser'] = df_processed['age'] + df_processed['smoking_status_smokes'] + df_processed['bmi']
df_processed['age_risk_weighted_score'] = df_processed['age_glucose'] + df_processed['age_hypertension']

# Create the differents partitions for the model
x_train, x_test, y_train, y_test = prepared_data(df_processed, 'stroke')