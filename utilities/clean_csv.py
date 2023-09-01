import pandas as pd

# Load the data
data = pd.read_csv('smash_tourney_data.csv')

# Remove all rows with missing stage names
data = data[data['Stage Name'].notna()]


stages_to_remove = ['Unova Pokémon League', 'Castle Siege', 'WarioWare, Inc.', 'Frigate Orpheon', 'Skyloft', 'Magicant', 'Mario Circuit',
                    'Northern Cave', 'Mementos', 'Yggdrasil’s Altar']
data = data[~data['Stage Name'].isin(stages_to_remove)]

# Export as a new CSV file
data.to_csv('../data/cleaned_smash_tourney_data.csv', index=False)