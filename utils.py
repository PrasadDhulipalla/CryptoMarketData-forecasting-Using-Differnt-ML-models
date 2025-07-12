import matplotlib.pyplot as plt

def plot_price(df):
    df['Close'].plot(figsize=(12, 6), title="Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.show()
