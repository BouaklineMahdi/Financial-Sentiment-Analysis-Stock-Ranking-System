import pandas as pd
import yfinance as yf
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

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

def fetch_stock_data(symbol):
    try:
        historical_data = download_historical_data(symbol)
        financial_data = fetch_financial_data(symbol)
        additional_info = fetch_additional_info(symbol)  # Fetch additional info using yfinance
        return historical_data, financial_data, additional_info
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

def fetch_additional_info(symbol):
    try:
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




# ... (previous code remains unchanged)

def main():
    # Fetch and store data in MongoDB
    sp500_symbols = fetch_sp500_symbols()
    if not sp500_symbols:
        print("No S&P 500 symbols found. Exiting.")
        exit()

    db = client["historic_data"]  # Replace with your database name
    features, targets = [], []

    


# Make sure to call the main function
if __name__ == "__main__":
    main()