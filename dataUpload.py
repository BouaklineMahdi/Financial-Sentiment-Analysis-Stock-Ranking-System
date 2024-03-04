import numpy
import pandas as pd
import yfinance

def main():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' #get our data on current S&P 500 companies
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0] 

    sp500_symbols = sp500_df['Symbol'].tolist()

    for ticker in sp500_symbols:
        ticker_info = yfinance.Ticker(ticker).info
        ticker_info.get('marketCap', None)
        ticker_info.get('regularMarketPrice', None)
        ticker_info.get('averageVolume', None)
        ticker_info.get('dividendYield', None)
        ticker_info.get('trailingEps', None)
        ticker_info.get('trailingPE', None)
        ticker_info.get('revenue', None)
        ticker_info.get('netIncome', None)
        


if __name__ == "__main__":
    main()