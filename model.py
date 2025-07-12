from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(df):
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    joblib.dump(model, "models/trained_model.pkl")

    return model, mse
