
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    # Load processed data
    df = pd.read_csv('data/processed_data.csv')

    # Define features and target
    features = ['Age', 'Gender', 'Location', 'AccountAge', 'Frequency', 'AvgBasketSize', 'Promotions', 'Recency']
    target = 'PurchaseHistory'  # CLV prediction
    
    X = df[features]
    y = df[target]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    print("Model Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")

    return model

# Train and save the model
model = train_model()
