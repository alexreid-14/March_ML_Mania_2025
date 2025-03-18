import pandas as pd
import numpy as np 
import os 
from pathlib import Path

pd.set_option('display.max_columns', None)

output = [ 
    'School', 'SRS', 'SOS', 'Pace', 'FTr', '3PAr'
]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through the years from 2025 to 2014
for year in range(2003, 2026):
    # Read the CSV file for the current year
    df = pd.read_csv(f"../../data/external/{year}_adv_stats.csv")
    
    # Select the specified columns
    df = df[output]
    
    # Add the Season column
    df['Season'] = year
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all the DataFrames together
all_seasons_df = pd.concat(dfs, ignore_index=True)

# Check if 'School' ends with ' NCAA' and remove it
all_seasons_df['School'] = all_seasons_df['School'].apply(lambda x: x.rstrip('NCAA') if x.endswith('NCAA') else x)
all_seasons_df['School'] = all_seasons_df['School'].str.strip()
all_seasons_df['School'] = all_seasons_df['School'].apply(lambda x: 'UCLA' if x == 'UCL' else x)
#all_seasons_df.loc[(advanced_stats['Season'] == 2004) & (all_seasons_df['School'].str.contains('utsa', case=False)), 'School'] = 'UTSA'


output_dir = Path("../..") / "data" / "preprocessing"
output_dir.mkdir(parents=True, exist_ok =True)
output_path = output_dir / "mens_advanced_stats.csv"
all_seasons_df.to_csv(output_path, index=False)
print(f"File successfully exported to {output_path}")
