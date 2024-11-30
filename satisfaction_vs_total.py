import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

def feature_processing(df):
    df['expected_draws'] = df['expected_draws'] / df['guaranteed_draws']
    df['available_draws'] = df['available_draws'] / df['guaranteed_draws']
    df['already_draws'] = df['already_draws'] / df['guaranteed_draws']
    df['total_draws'] = df['total_draws'] / df['guaranteed_draws']
    df['guaranteed_draws'] = df['guaranteed_draws'] / df['guaranteed_draws'] 
    df['got_character'] = df['got_character'].apply(lambda x: 0 if x !=1 else 1)
    
    return df

def average_data(df):
    numeric_columns = ['expected_draws', 'available_draws', 'already_draws', 'guaranteed_draws', 'total_draws', 'satisfaction']
    average_values = df[numeric_columns].mean().round().astype(int)

    categorical_columns = ['preference', 'got_character']
    mode_values = df[categorical_columns].mode().iloc[0]

    average_player = pd.concat([average_values, mode_values])
    
    return average_player

data = pd.read_csv('data.csv')
average_player = average_data(data)

data = feature_processing(data)
X = data[['expected_draws', 'available_draws', 'preference', 'already_draws', 'guaranteed_draws', 'total_draws', 'got_character']]
y = data['satisfaction']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

new_case = {
    'expected_draws': average_player['expected_draws'],
    'available_draws': average_player['available_draws'],
    'preference': average_player['preference'],
    'already_draws': average_player['already_draws'],
    'guaranteed_draws': average_player['guaranteed_draws']
}
 
guaranteed_draws = new_case['guaranteed_draws']

total_draws_percentages = np.linspace(0.1, 1.0, 40) 
total_draws_points = (total_draws_percentages * guaranteed_draws).astype(int) 
predicted_satisfaction = []

for total_draws in total_draws_points:
    got_character = 1
    new_data = pd.DataFrame({
        'expected_draws': [new_case['expected_draws']],
        'available_draws': [new_case['available_draws']],
        'preference': [new_case['preference']],
        'already_draws': [new_case['already_draws']],
        'guaranteed_draws': [new_case['guaranteed_draws']],
        'total_draws': [total_draws],
        'got_character': [got_character]
    })
    new_data = feature_processing(new_data)
    satisfaction = model.predict(new_data)[0]
    predicted_satisfaction.append(satisfaction)

plt.figure(figsize=(10, 6))
plt.plot(total_draws_percentages * 100, predicted_satisfaction, '-o', label='Predicted Satisfaction', color='blue')
plt.xlabel('Total Draws as % of Guaranteed Draws', fontsize=12)
plt.ylabel('Predicted Satisfaction', fontsize=12)
plt.title('Satisfaction vs Total Draws Percentage', fontsize=14)
plt.grid(True)
plt.axvline(x=(new_case['expected_draws'] / guaranteed_draws) * 100, color='red', linestyle='--', label='Expected Draws')
plt.legend()

if os.path.exists("satisfaction_vs_total.png"):
    os.remove("satisfaction_vs_total.png")
plt.savefig("satisfaction_vs_total.png")
plt.show()

