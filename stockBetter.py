import numpy as np
import pandas as pd
import yfinance
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import interpolate

uri = "mongodb+srv://user:Chino@cluster0.tbowifi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    print("Connecting to MongoDB...")
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')  # Test if the connection is successful
    print("Connected to MongoDB successfully!")

    db = client["historic_data"] 
    print("Connected to historic_data database.")

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit()

def preprocess_data(historical_data):
    try:
        print("Preprocessing data...")
        X_2 = np.array([[d.get('Open'), d.get('High'), d.get('Low'), d.get('Volume')] for d in historical_data])

        y = np.array([d.get("Close") for d in historical_data])

        X_2 = np.nan_to_num(X_2)
        
        # Convert all columns to numeric
        X_2 = pd.DataFrame(X_2).apply(pd.to_numeric, errors='coerce')

        # Interpolate NaN values
        X_2 = X_2.interpolate(method='linear', axis=0).values

        # Check if X_2 is numeric data type
        if not np.issubdtype(X_2.dtype, np.number):
            print("X_2 is not a numeric data type.")
            return None, None, None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Failed to preprocess: {e}")
        return None, None, None, None  # Return four None values instead of a single None


def train_model(features, targets):
    print("Training the model...")

    # Convert features and targets to float type
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)

    

    print(f"Debug: Clean Features shape: {features.shape}")
    print(f"Debug: Clean Targets shape: {targets.shape}")

    # Model structure
    model = Sequential([
        Dense(64, activation='relu', input_shape=(features.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping configuration
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model training
    history = model.fit(features, targets, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    print("Debug: Model training completed.")
    print("Debug: Training history:", history.history)

    return model



# Rank companies based on the model's predictions
def rank_companies(model, features, sp500_symbols):
    try:
        print("Ranking companies based on predictions...")
        predictions = model.predict(features)
        return pd.DataFrame({"Symbol": sp500_symbols, "Prediction": predictions.flatten()}).sort_values(by="Prediction", ascending=False).head(5)
    except Exception as e:
        print(f"Failed to rank companies: {e}")
        return pd.DataFrame()
    
def captureData(ind):
    data = db[ind].find()
    recorddata = []
    for record in data:
        record.pop('_id')
        if (recorddata is not None):
            recorddata.append(record)    
    return recorddata
    
def main():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_symbols = sp500_df['Symbol'].tolist()
    x_train_vals = []
    x_test_vals = []
    y_train_vals = []
    y_test_vals = []
    for i in sp500_symbols:
        historicalrecords = captureData(i)
        if (len(historicalrecords)==0):
            continue
        additionalinfo = historicalrecords[len(historicalrecords)-1]
        historicalrecords.pop(-1)
        x_train, x_test, y_train, y_test = preprocess_data(historicalrecords,additionalinfo)
        x_train_vals.append(x_train)
        x_test_vals.append(x_test)
        y_train_vals.append(y_train)
        y_test_vals.append(y_test)
    
    x_test = np.vstack(x_test_vals)


        
        


if __name__ == "__main__":
    main()