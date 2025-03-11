
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Handle missing values (simple approach, can be enhanced)
    df.fillna(df.mean(), inplace=True)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['Location'] = label_encoder.fit_transform(df['Location'])

    # Feature engineering
    df['Recency'] = (pd.to_datetime('today') - pd.to_datetime(df['LastPurchaseDate'])).dt.days
    df.drop(columns=['LastPurchaseDate'], inplace=True)  # Drop the original last purchase date

    return df

# Preprocess data and save
df = preprocess_data('data/raw_data.csv')
df.to_csv('data/processed_data.csv', index=False)
