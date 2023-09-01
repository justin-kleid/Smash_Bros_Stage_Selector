import pandas as pd

data = pd.read_csv('../data/cleaned_smash_tourney_data.csv')
print(data['Stage Name'].value_counts())