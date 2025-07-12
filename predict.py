import joblib

def predict_next(df):
    model = joblib.load("models/trained_model.pkl")
    latest = df.iloc[-1][['Open', 'High', 'Low', 'Volume']].values.reshape(1, -1)
    prediction = model.predict(latest)
    return prediction[0]
