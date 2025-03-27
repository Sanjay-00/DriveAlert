import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define the path to the CSV file
file_path = 'C:/Users/91748/Desktop/new.csv'  # Replace with the path to your CSV file

# Read the CSV file to get the list of stock symbols and their types
symbols_df = pd.read_csv(file_path)
trackers = symbols_df[['Symbol', 'Type']].dropna()  # Drop rows with NaN values

# Ensure 'Symbol' column has no non-string values
trackers = trackers[trackers['Symbol'].apply(lambda x: isinstance(x, str))]

# Get the current date and the date 365 days before today
now = datetime.now()
start_date = datetime(2010, 1, 1)



# Format the dates for Yahoo Finance
start_date_str = start_date.strftime('%Y-%m-%d')  # Example: 2023-09-14
end_date_str = now.strftime('%Y-%m-%d')  # Example: 2024-09-13

# Initialize a list to hold all data
data_frames = []

for _, tracker in trackers.iterrows():
    symbol = tracker['Symbol'].upper()
    type_of_asset = tracker['Type']
    
    try:
        # Fetch data from Yahoo Finance
        df = yf.download(symbol, start=start_date_str, end=end_date_str, interval='1d')
        df['Symbol'] = symbol  # Add a column for the stock symbol
        df['Type'] = type_of_asset  # Add a column for the type (Stock or Crypto)
        df.reset_index(inplace=True)  # Reset index to have 'Date' as a column
        data_frames.append(df)  # Append the DataFrame to the list
        print(f""Data for {symbol} downloaded successfully."")
    except Exception as e:
        print(f""Failed to download data for {symbol}: {e}"")

# Concatenate all DataFrames into a single DataFrame
if data_frames:
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(""Columns in combined data:"", combined_data.columns)

    # Reorder columns for better readability
    combined_data = combined_data[['Symbol', 'Type', 'Date', 'Open', 'High', 'Low', 'Close','Adj Close', 'Volume']]
    
    # Display the combined DataFrame
    print(combined_data.head())
else:
    print(""No data to combine."")
