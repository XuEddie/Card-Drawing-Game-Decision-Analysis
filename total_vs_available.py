import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('data.csv')

data['available_draws_ratio'] = data['available_draws'] / data['guaranteed_draws']
data['total_draws_ratio'] = data['total_draws'] / data['guaranteed_draws']

plt.figure(figsize=(10, 6))
plt.scatter(data['available_draws_ratio'], data['total_draws_ratio'], alpha=0.7, color='blue', label='Player Data')

x = [0, max(data['available_draws_ratio'].max(), data['total_draws_ratio'].max())]
plt.plot(x, x, color='red', linestyle='--', label='y = x (Risk Threshold)')

plt.xlabel('Available Draws Ratio(Available/Guaranteed)', fontsize=12)
plt.ylabel('Total Draws Ratio(Total/Guaranteed)', fontsize=12)
plt.title('Player Risk Analysis: Total Draws vs. Available Draws', fontsize=14)
plt.legend()
plt.grid()

if os.path.exists("total_vs_available.png"):
    os.remove("total_vs_available.png")
plt.savefig("total_vs_available.png")
plt.show()
