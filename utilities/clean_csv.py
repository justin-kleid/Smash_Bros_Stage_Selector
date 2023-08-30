import pandas as pd

# Load the data
data = pd.read_csv('smash_tourney_data.csv')

# Remove all rows with missing stage names
data = data[data['Stage Name'].notna()]

# Export as a new CSV file
data.to_csv('cleaned_smash_tourney_data.csv', index=False)