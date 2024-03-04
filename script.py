import pandas as pd
import yfinance as yf
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# MongoDB connection URI
uri = "mongodb+srv://user:Chino@cluster0.tbowifi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')  # Test if the connection is successful
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit()

# Fetch S&P 500 symbols
def fetch_sp500_symbols():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_table = pd.read_html(url)
        sp500_df = sp500_table[0]
        sp500_symbols = sp500_df['Symbol'].tolist()
        return sp500_symbols
    except Exception as e:
        print(f"Failed to fetch S&P 500 symbols: {e}")
        return []

# Download historical data for a symbol
def download_historical_data(symbol, start_date='2024-01-01', end_date='2024-03-01'):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Failed to download historical data for {symbol}: {e}")
        return pd.DataFrame()

# Fetch financial data for a symbol
def fetch_financial_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        financials = ticker.financials
        return financials
    except Exception as e:
        print(f"Failed to fetch financial data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_additional_info(symbol):
    try:
        # Fetch additional info using yfinance
        ticker_info = yf.Ticker(symbol).info

        return {
            'longName': ticker_info.get('longName', None),
            'industry': ticker_info.get('industry', None),
            'marketCap': ticker_info.get('marketCap', None),
            'regularMarketPrice': ticker_info.get('regularMarketPrice', None),
            'averageVolume': ticker_info.get('averageVolume', None),
            'dividendYield': ticker_info.get('dividendYield', None),
            'trailingEps': ticker_info.get('trailingEps', None),
            'trailingPE': ticker_info.get('trailingPE', None),
            'revenue': ticker_info.get('revenue', None),
            'netIncome': ticker_info.get('netIncome', None)
        }
    except Exception as e:
        print(f"Failed to fetch additional info for {symbol}: {e}")
        return {}

def fetch_stock_data(symbol):
    try:
        historical_data = download_historical_data(symbol)
        financial_data = fetch_financial_data(symbol)
        additional_info = fetch_additional_info(symbol)  # Fetch additional info using yfinance
        return historical_data, financial_data, additional_info
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

def store_data_in_mongodb(symbol, stock_data, financial_data, additional_info):
    try:
        db = client["historic_data"]  # Replace with your database name
        collection = db[symbol]

        # Convert Timestamp index to string on a copy of the DataFrame
        stock_data_copy = stock_data.copy()
        stock_data_copy.index = stock_data_copy.index.strftime('%Y-%m-%d %H:%M:%S')

        # Convert DataFrame to list of dictionaries
        stock_data_dict = stock_data_copy.reset_index().to_dict("records")
        stock_data_dict = [{str(k): v for k, v in item.items()} for item in stock_data_dict]

        # Convert financial_data DataFrame to a dictionary
        financial_data_dict = financial_data.to_dict()
        financial_data_dict = {str(k): v for k, v in financial_data_dict.items()}

        # Combine stock_data, financial_data, and additional_info into a single dictionary for insertion
        data_to_insert = stock_data_dict + [{"financials": financial_data_dict, "additional_info": additional_info}]

        # Use insert_many for bulk insertion
        collection.insert_many(data_to_insert)
        print(f"Data for {symbol} stored in MongoDB successfully!")
    except Exception as e:
        print(f"Failed to store data for {symbol} in MongoDB: {e}")

# Define and train the AI model
def train_model(features, targets):
    try:
        # Check for NaN values in features and targets
        if any(pd.isna(features).any()) or any(pd.isna(targets).any()):
            raise ValueError("NaN values detected in features or targets. Check your data.")

        model = Sequential([
            Dense(64, activation='relu', input_shape=(features.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Add early stopping to prevent NaN values in the loss
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(features, targets, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        return model
    except Exception as e:
        print(f"Failed to train the model: {e}")
        return None

# Rank companies based on the model's predictions
def rank_companies(model, features, sp500_symbols):
    try:
        predictions = model.predict(features)
        return pd.DataFrame({"Symbol": sp500_symbols, "Prediction": predictions.flatten()}).sort_values(by="Prediction", ascending=False).head(5)
    except Exception as e:
        print(f"Failed to rank companies: {e}")
        return pd.DataFrame()

def preprocess_data(stock_data):
    try:
        scaler = MinMaxScaler()
        stock_data['Close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        # Create features (X) and targets (y)
        X = stock_data['Close'].values[:-1].reshape(-1, 1)
        y = stock_data['Close'].values[1:].reshape(-1, 1)

        # Ensure X and y have the same length
        min_length = min(len(X), len(y))

        if min_length == 0:
            raise ValueError("Length of X or y is zero.")

        X = X[:min_length]
        y = y[:min_length]

        return X, y
    except Exception as e:
        print(f"Failed to preprocess data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ... (previous code remains unchanged)

def main():
    # Fetch and store data in MongoDB
    sp500_symbols = fetch_sp500_symbols()
    if not sp500_symbols:
        print("No S&P 500 symbols found. Exiting.")
        exit()

    db = client["historic_data"]  # Replace with your database name
    features, targets = [], []

    for symbol in sp500_symbols:
        # Always attempt to store data in the collection
        historical_data, financial_data, additional_info = fetch_stock_data(symbol)
        if historical_data.empty or financial_data.empty:
            print(f"No valid data found for {symbol}. Skipping to the next symbol.")
            continue  # Skip storing data for this symbol

        store_data_in_mongodb(symbol, historical_data, financial_data, additional_info)

        # Retrieve data from MongoDB for the current symbol
        stock_data = pd.DataFrame(list(db[symbol].find()))
        if stock_data.empty:
            print(f"Data is missing for {symbol}. Skipping to the next symbol.")
            continue  # Skip processing this symbol if data is missing

        # Preprocess data for the current symbol
        features_i, targets_i = preprocess_data(stock_data)
        if len(features_i) != len(targets_i):
            print(f"Invalid data length for {symbol}. Skipping to the next symbol.")
            continue  # Skip processing this symbol if features and targets have different lengths
        features.append(features_i)
        targets.append(targets_i)

    # Check if both features and targets lists are empty
    if not features or not targets:
        print("No valid data found for any S&P 500 symbols. Exiting.")
        exit()

    # Concatenate features and targets for all symbols
    features = pd.concat([pd.DataFrame(arr) for arr in features])
    targets = pd.concat([pd.DataFrame(arr) for arr in targets])

    # Train the model
    model = train_model(features, targets)
    if model is None:
        print("Failed to train the model. Exiting.")
        exit()

    # Rank companies based on the model's predictions
    try:
        ranked_companies = rank_companies(model, features, sp500_symbols)
        if not ranked_companies.empty:
            print("Top 5 Companies to Invest In:")
            print(ranked_companies)
        else:
            print("Failed to rank companies. No results.")
    except Exception as rank_error:
        print(f"Failed to rank companies: {rank_error}")
        exit()

# Make sure to call the main function
if __name__ == "__main__":
    main()