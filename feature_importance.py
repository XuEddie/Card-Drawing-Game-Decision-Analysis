import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

data = pd.read_csv('data.csv')

def feature_processing(df):
    df['expected_draws'] = df['expected_draws'] / df['guaranteed_draws']
    df['available_draws'] = df['available_draws'] / df['guaranteed_draws']
    df['already_draws'] = df['already_draws'] / df['guaranteed_draws']
    df['total_draws'] = df['total_draws'] / df['guaranteed_draws']
    df['got_character'] = df['got_character'].apply(lambda x: 0 if x != 1 else 1)
    return df

data = feature_processing(data)
X = data[['expected_draws', 'available_draws', 'preference', 
          'already_draws', 'guaranteed_draws', 'total_draws', 'got_character']]
y = data['satisfaction']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', orient='h')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance for Satisfaction Prediction', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

if os.path.exists("feature_importance.png"):
    os.remove("feature_importance.png")
plt.savefig("feature_importance.png")
plt.show()
