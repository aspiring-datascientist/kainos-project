
KAINOS Project: Forecasting Inventory for Shows

 Overview

This project aims to forecast the remaining ticket availability for events using a Decision Tree Regressor. The dataset contains information on events, including event dates, ticket availability, and timestamps for when the data was collected.

 Project Structure

KAINOS_project/

cleaned_data.csv          # Cleaned dataset used for modeling
requirements.txt          # Dependencies for the project
notebook.ztnb             # Jupyter Notebook containing the analysis and model
README.md                 # Project documentation


Dependencies

This project requires the following Python packages:


 numpy
 pandas
 matplotlib
 scikit-learn
 zero-true >= 0.4.3

To install the required packages, run:

bash
pip install -r requirements.txt


 Getting Started

1. Set Up Environment:
   - Ensure you have Python 3.8 or greater installed.
   - Create a new virtual environment and activate it.

2. Run Zero-True:
   - Navigate to the project directory:
     bash
     cd path/to/KAINOS_project
   
   - Start the Zero-True environment:
     bash
     zero-true
     

3. Open the Notebook:
   - Access the Jupyter Notebook interface to run the analysis and model:
   - Open `notebook.ztnb`.

 Usage

- The main objective of the analysis is to predict the `remaining_tickets` based on the number of `days_until_event`.
- The model is evaluated using Mean Squared Error (MSE) to measure prediction accuracy.

 Code

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

 Load data
df = pd.read_csv('cleaned_data.csv')

 Preprocess data
df['event_date'] = pd.to_datetime(df['event_date'], utc=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y_%m_%dT%H_%M_%S', utc=True)
df['days_until_event'] = (df['event_date'] - df['timestamp']).dt.days
df['remaining_tickets'] = df['availability_standard'] + df['availability_resale']

 Train-test split
X = df[['days_until_event', 'remaining_tickets']]
y = df['remaining_tickets']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

 Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

 Plot actual vs predicted with '+' markers and a diagonal line
plt.figure(figsize=(10, 6))

 Scatter plot for actual and predicted values
plt.scatter(X_test['days_until_event'], y_test, label='Actual', color='blue', s=50, alpha=0.7, marker='+')
plt.scatter(X_test['days_until_event'], y_pred, label='Predicted', color='red', s=50, alpha=0.7, marker='+')

 Add diagonal line for perfect predictions
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='Ideal (y = y_pred)')
plt.yticks(np.arange(0, max_val + 400, 400))
plt.xticks(np.arange(0, max_val + 30, 30))

plt.xlabel('Days Until Event')
plt.ylabel('Remaining Tickets')

plt.xlim([0, 365])
plt.ylim([0, 2000])

plt.legend()
plt.title('Actual vs Predicted Remaining Tickets with Ideal Diagonal')
plt.grid(True)
plt.tight_layout()
plt.show()


Publishing

To publish the project on Zero-True, ensure you have all necessary files in the project directory and follow these steps:

1. Make sure the Zero-True environment is running.
2. From the command line in the project directory, use:
   bash
   zero-true publish
  

 Conclusion

This project provides a foundation for forecasting ticket availability for events using machine learning techniques. Further enhancements can be made by exploring other models or refining the dataset.