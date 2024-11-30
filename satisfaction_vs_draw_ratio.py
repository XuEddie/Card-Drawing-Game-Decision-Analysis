import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data = pd.read_csv('data.csv')

filtered_data = data[data['got_character'] == 1]

filtered_data.loc[:, 'draw_ratio'] = filtered_data['total_draws'] / filtered_data['expected_draws']

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['draw_ratio'], filtered_data['satisfaction'], color='blue', alpha=0.7, label='Samples (got_character=1)')
plt.xlabel('Total Draws / Expected Draws', fontsize=12)
plt.ylabel('Satisfaction', fontsize=12)
plt.title('Satisfaction vs Total Draws / Expected Draws (got_character=1)', fontsize=14)
plt.grid(True)
plt.axvline(x=1, color='red', linestyle='--', label='Expected Draws Ratio = 1')
plt.legend()

if os.path.exists("satisfaction_vs_draw_ratio.png"):
    os.remove("satisfaction_vs_draw_ratio.png")
plt.savefig("satisfaction_vs_draw_ratio.png")
plt.show()
