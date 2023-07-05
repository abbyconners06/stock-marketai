import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('AAPL.csv')

# Prepare the data
X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
y = df['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for tomorrow's stock price
tomorrow_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values.reshape(1, -1)
predicted_price = model.predict(tomorrow_data)

print('Predicted stock price for tomorrow:', predicted_price)