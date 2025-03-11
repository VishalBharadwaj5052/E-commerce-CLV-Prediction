
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('clv_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Preprocessing
    df['Recency'] = (pd.to_datetime('today') - pd.to_datetime(df['LastPurchaseDate'])).dt.days
    df.drop(columns=['LastPurchaseDate'], inplace=True)
    
    # Predict CLV
    prediction = model.predict(df)
    return jsonify({'predicted_clv': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
