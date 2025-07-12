from src.data_loader import load_data
from src.utils import plot_price
from src.model import train_model
from src.predict import predict_next

def main():
    df = load_data("data/BTC-USD.csv")
    plot_price(df)
    model, mse = train_model(df)
    print(f"Model trained. MSE: {mse:.2f}")
    next_price = predict_next(df)
    print(f"Predicted next closing price: ${next_price:.2f}")

if __name__ == "__main__":
    main()
