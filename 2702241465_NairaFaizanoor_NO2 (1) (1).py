#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.encoders = {}

    def handle_missing_values(self):
        """Fill missing values with the mean of the column."""
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                mean_value = self.df[column].mean()
                self.df[column] = self.df[column].fillna(mean_value)

    def encode_categorical(self, categorical_columns):
        """Convert categorical columns to label encoding or ordinal encoding."""
        for col in categorical_columns:
            if self.df[col].dtype == 'object':
                if col == 'person_education':
                    ordinal_encoder = OrdinalEncoder(categories=[['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']])
                    self.df[col] = ordinal_encoder.fit_transform(self.df[[col]])
                    self.encoders[col] = ordinal_encoder
                else:
                    label_encoder = LabelEncoder()
                    self.df[col] = label_encoder.fit_transform(self.df[col])
                    self.encoders[col] = label_encoder

    def split_data(self, target_column, test_size=0.2):
        """Split the data into training and testing sets."""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def save_encoders(self, encoder_filenames):
        """Save the encoders used for categorical features."""
        for col, encoder in self.encoders.items():
            filename = encoder_filenames.get(col, f'{col}_encoder.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(encoder, f)
            print(f"{col} encoder saved as {filename}")

class XGBoostTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train(self):
        """Train the XGBoost model."""
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def save_model(self, filename='xgboost_model.pkl'):
        """Save the trained model using pickle."""
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"XGBoost model saved as {filename}")

# Load dataset
df = pd.read_csv('Dataset_A_loan.csv')

# Data Preprocessing
preprocessor = DataPreprocessor(df)
preprocessor.handle_missing_values()
preprocessor.encode_categorical(['person_gender', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file', 'person_education'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = preprocessor.split_data(target_column='loan_status')

# Train the model using XGBoost
trainer = XGBoostTrainer(X_train, X_test, y_train, y_test)
trainer.train()

# Save the trained model
trainer.save_model()

# Save the encoders
encoder_filenames = {
    'person_gender': 'label_encoder_gender.pkl',
    'previous_loan_defaults_on_file': 'label_encoder_defaults.pkl',
    'person_home_ownership': 'label_encoder_home.pkl',
    'loan_intent': 'label_encoder_loan_intent.pkl',
    'person_education': 'ordinal_encoder_education.pkl'
}
preprocessor.save_encoders(encoder_filenames)


# In[ ]:




