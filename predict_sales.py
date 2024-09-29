import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the cleaned data from the CSV file
df_clean = pd.read_csv('cleaned_data.csv')

# Convert 'timestamp' and 'event_date' into datetime format
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean['event_date'] = pd.to_datetime(df_clean['event_date'])

# Drop columns that aren't useful for prediction or are non-numeric
df_clean = df_clean.drop(columns=['timestamp', 'event_date'])

# Encode categorical columns (like 'section_id' and 'venue_id')
label_encoder = LabelEncoder()
df_clean['section_id'] = label_encoder.fit_transform(df_clean['section_id'])
df_clean['venue_id'] = label_encoder.fit_transform(df_clean['venue_id'])

# Use 'availability_standard' as the target variable (what you want to predict)
X = df_clean.drop(columns=['availability_standard'])  # Features
y = df_clean['availability_standard']  # Target

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
error = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", error)

# Save the trained model
joblib.dump(model, 'ticket_sales_model.pkl')
