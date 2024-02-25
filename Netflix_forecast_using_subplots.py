# Importing Necessay Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# reading the data
data = pd.read_csv('Netflix-Subscriptions.csv')
print(data.head())

data['Time Period'] = pd.to_datetime(data['Time Period'], 
                                     format='%d/%m/%Y')
print(data.head())

#Subplot
fig = plt.figure(figsize=(12,6))

# Add the first subplot (top-left)
axs1 = fig.add_subplot(1, 3, 1)
axs1.plot(data['Time Period'],data['Subscribers'],label='Subscribers')
          #linestyle="dotted")
axs1.set_title("Quarterly Subscriptions Growth")
axs1.set_xlabel("Date")
axs1.set_ylabel("Netflix Subscriptions")
axs1.legend()


# Calculate the quarterly growth rate
data['Quarterly Growth Rate'] = data['Subscribers'].pct_change() * 100

# Create a new column for bar color (green for positive growth, red for negative growth)
data['Bar Color'] = data['Quarterly Growth Rate'].apply(lambda x: 'green' if x > 0 else 'red')

# Plot the quarterly growth rate using bar graphs
#Subplot
axs2 = fig.add_subplot(1, 3, 2)
axs2.bar(data['Time Period'],data['Quarterly Growth Rate'],color=data['Bar Color'])
axs2.set_title("Quarterly Subscriptions Growth Rate")
axs2.set_xlabel("Time Period")
axs2.set_ylabel("Quarterly Growth Rate (%)")



# Calculate the yearly growth rate
data['Year'] = data['Time Period'].dt.year
yearly_growth = data.groupby('Year')['Subscribers'].pct_change().fillna(0) * 100

# Create a new column for bar color (green for positive growth, red for negative growth)
data['Bar Color'] = yearly_growth.apply(lambda x: 'green' if x > 0 else 'red')


axs3 = fig.add_subplot(1, 3, 3)
axs3.bar(data['Year'],yearly_growth,color=data['Bar Color'])
axs3.set_title("Netflix Yearly Subscriptions Growth Rate")
axs3.set_xlabel("Year")
axs3.set_ylabel("Yearly Growth Rate (%)")

time_series = data.set_index('Time Period')['Subscribers']

differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

p, d, q = 1, 1, 1
model = ARIMA(time_series, order=(p, d, q))
results = model.fit()
print(results.summary())


future_steps = 5
predictions = results.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)

# Create a DataFrame with the original data and predictions
forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

# Plot the original data and prediction
plt.plot(forecast.index, forecast['Predictions'], label='Predictions')
plt.plot(forecast.index, forecast['Original'], label='Original Data')

plt.title('Netflix Quarterly Subscription Predictions')
plt.xlabel('Time Period')
plt.ylabel('Subscribers')
plt.legend()
plt.show()

