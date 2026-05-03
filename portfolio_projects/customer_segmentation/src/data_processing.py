import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Customer_Segmentation_Using_K-Means_Clustering/data/OnlineRetail.csv'
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Adjust encoding if needed
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

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

# Print summary statistics for the cleaned data
print("\nSummary Statistics of Cleaned Data:")
print(df.describe(include='all'))

# Convert 'InvoiceDate' to datetime format
try:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
except ValueError:
    print("\nValueError encountered during date conversion. Trying with `dayfirst=True`")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

print("\nData Types After Conversion:")
print(df.dtypes)

# Calculate RFM metrics
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

rfm_df.reset_index(inplace=True)

print("\nRFM Data Shape:", rfm_df.shape)
print("Preview of RFM Data:")
print(rfm_df.head())

# Print summary statistics for the RFM data
print("\nSummary Statistics of RFM Data:")
print(rfm_df.describe())

# Normalize the RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])

# Add the normalized features back to the DataFrame
rfm_df[['Recency_scaled', 'Frequency_scaled', 'MonetaryValue_scaled']] = rfm_scaled

print("\nRFM Data with Normalized Features:")
print(rfm_df.head())

# Ensure the 'data' directory exists
output_dir = '/Users/paigeleeseberg/Downloads/Python-Projects/Customer_Segmentation_Using_K-Means_Clustering/data'
os.makedirs(output_dir, exist_ok=True)

# Save the RFM data with normalized features to CSV
try:
    rfm_df.to_csv(os.path.join(output_dir, 'rfm_clusters.csv'), index=False)
    print("\nRFM data with normalized features saved to 'data/rfm_clusters.csv'.")
except Exception as e:
    print(f"Error saving file: {e}")
