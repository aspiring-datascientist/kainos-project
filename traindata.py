import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv(r'C:\Users\HP\one drive pastages\Desktop\kainos\cleaned_data.csv')

# Step 2: Data Preprocessing
# Convert event_date and timestamp to datetime format
df['event_date'] = pd.to_datetime(df['event_date'], utc=True)  # Ensure timezone-aware (UTC)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y_%m_%dT%H_%M_%S', utc=True)  # Ensure timezone-aware (UTC)

# Step 3: Feature Engineering
# Calculate Days Until Event
df['days_until_event'] = (df['event_date'] - df['timestamp']).dt.days

# Add other useful features like availability and demand
df['demand'] = df['capacity'] - (df['availability_standard'] + df['availability_resale'])  # Inverse of availability
df['remaining_tickets'] = df['availability_standard'] + df['availability_resale']

# Step 4: Exploratory Data Analysis (EDA)
# Plot Remaining Tickets vs. Days Until Event for different sections
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='days_until_event', y='remaining_tickets', hue='section_id', palette="tab10")
plt.title('Remaining Tickets by Section Over Time')
plt.xlabel('Days Until Event')
plt.ylabel('Remaining Tickets')
plt.grid(True)
plt.show()

# Step 5: Train-Test Split for Forecasting
X = df[['days_until_event', 'remaining_tickets', 'demand']]  # Features
y = df['remaining_tickets']  # Target: Remaining Tickets

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Decision Tree Regressor for Forecasting
tree_model = DecisionTreeRegressor(random_state=42)  # Set random_state for reproducibility
tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred = tree_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Step 7: Plot Predictions vs Actuals
plt.figure(figsize=(10, 6))
plt.scatter(X_test['days_until_event'], y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test['days_until_event'], y_pred, color='red', label='Predicted', alpha=0.6)
plt.title('Decision Tree: Predicted vs Actual Remaining Tickets')
plt.xlabel('Days Until Event')
plt.ylabel('Remaining Tickets')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Final Forecast Results
final_results = pd.DataFrame({
    'days_until_event': X_test['days_until_event'],
    'actual_remaining_tickets': y_test,
    'predicted_remaining_tickets': y_pred
})

print(final_results)