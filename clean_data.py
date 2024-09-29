import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')


# Check for missing data (optional, just to see where missing data exists)
print("Missing data in each column:\n", df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()

# Convert 'event_date' to datetime format (if it's not already in that format)
df_clean['event_date'] = pd.to_datetime(df_clean['event_date'], errors='coerce')

# Print the first 10 rows of the cleaned dataset
print(df_clean.head(10))

# Save the cleaned data to a new CSV file
df_clean.to_csv('cleaned_data.csv', index=False)


