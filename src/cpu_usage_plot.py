import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
df1 = pd.read_csv('tensorizer_loading.csv')  # Update the path to your first CSV file
df2 = pd.read_csv('hf_loading.csv')  # Update the path to your second CSV file

# Assuming the CSVs have columns 'Time (s)' and 'CPU Usage (%)'
plt.figure(figsize=(10, 6))

plt.plot(df1['Time (s)'], df1['CPU Usage (%)'], label='hf_loading')
plt.plot(df2['Time (s)'], df2['CPU Usage (%)'], label='Tensorizer')

plt.title('CPU Usage Comparison - /runpod-volume/model-7b')
plt.xlabel('Time (s)')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True)

plt.savefig('cpu_usage_comparison.png')

