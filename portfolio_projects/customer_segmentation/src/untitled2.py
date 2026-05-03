import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Customer_Segmentation_Using_K-Means_Clustering/data/OnlineRetail.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Adjust encoding if needed

print("Initial Data Shape:", df.shape)
print("Initial Data Preview:")
print(df.head())

# Clean the data
# Handle missing values
df = df.dropna(subset=['CustomerID', 'InvoiceDate', 'UnitPrice', 'Quantity'])
df['Description'].fillna('Unknown', inplace=True)

print("\nData Shape After Dropping Missing Values:", df.shape)
print("Preview of Data After Cleaning:")
print(df.head())

# Convert 'InvoiceDate' to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("\nData Types After Conversion:")
print(df.dtypes)

# Calculate RFM metrics
# Get the most recent date in the dataset
current_date = df['InvoiceDate'].max()

print("\nMost Recent Date in Dataset:", current_date)

# Create TotalSpend column
df['TotalSpend'] = df['UnitPrice'] * df['Quantity']

# Calculate RFM metrics
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalSpend': 'sum'  # Monetary Value
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'MonetaryValue'})

# Reset index to make 'CustomerID' a column again
rfm_df.reset_index(inplace=True)

print("\nRFM Data Shape:", rfm_df.shape)
print("Preview of RFM Data:")
print(rfm_df.head())

# Normalize the RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])

# Add the normalized features back to the DataFrame
rfm_df[['Recency_scaled', 'Frequency_scaled', 'MonetaryValue_scaled']] = rfm_scaled

print("\nRFM Data with Normalized Features:")
print(rfm_df.head())
