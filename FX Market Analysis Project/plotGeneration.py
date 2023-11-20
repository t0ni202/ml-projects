import pandas as pd
import joblib 
from google.colab import drive
import pickle

# Load the pickle file
drive.mount('/content/drive')
data_dict= pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/appml-assignment1-dataset-v2.pkl')

# Extract the variables X and y from the dictionary
X = data_dict['X']
y = data_dict['y']


# Load the pipeline
pipeline = joblib.load('pipeline1.pkl')

# Process the data
X_transformed = pipeline.transform(X)

# Create a new dataframe
X['day_of_week'] = X['date'].dt.dayofweek
X['trading_hour'] = X['date'].dt.hour
cad_change_df = X[['day_of_week', 'trading_hour']].copy()

# Calculate the difference between CAD-close and y (CAD-high from the next hour)
cad_change_df['cad_change'] = X['CAD-close'] - y

import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate root-mean-square
def rms(series):
    return np.sqrt(np.mean(series**2))

# Group by 'day_of_week' and 'trading_hour' and aggregate
aggregated = cad_change_df.groupby(['day_of_week', 'trading_hour'])['cad_change'].agg(
    mean='mean',
    percentile_5=lambda x: np.percentile(x, 5),
    percentile_10=lambda x: np.percentile(x, 10),
    percentile_25=lambda x: np.percentile(x, 25),
    percentile_50=lambda x: np.percentile(x, 50),
    percentile_75=lambda x: np.percentile(x, 75),
    percentile_90=lambda x: np.percentile(x, 90),
    percentile_95=lambda x: np.percentile(x, 95),
    rms=rms
).reset_index()


# Plotting
plt.figure(figsize=(15, 10))
for column in aggregated.columns[2:]:
    plt.plot(aggregated.index, aggregated[column], label=column)

plt.xlabel('(Day of Week, Hour of Day)')
plt.ylabel('cad-change value')
plt.title('Statistics of cad-change vs (Day of Week, Hour of Day)')
plt.legend()
plt.grid(True)
tick_locations = range(0, len(aggregated), 5)
labels = [f"{aggregated.iloc[i]['day_of_week']}, {aggregated.iloc[i]['trading_hour']}" for i in tick_locations]
plt.xticks(ticks=tick_locations, labels=labels,rotation=90)

plt.tight_layout()

plt.savefig('cad-change-stats.png')
plt.show()
