import pandas as pd
from itertools import combinations

# Load the team data
m_teams_df = pd.read_csv('../data/MTeams.csv')
w_teams_df = pd.read_csv('../data/WTeams.csv')
w_regular_season_df = pd.read_csv('../data/WRegularSeasonCompactResults.csv')

# Filter teams that are active in the year 2025
active_m_teams = m_teams_df[(m_teams_df['LastD1Season'] >= 2025) & (m_teams_df['FirstD1Season'] <= 2025)]
# Determine active women's teams based on WRegularSeasonCompactResults.csv
womens_2025 = w_regular_season_df[w_regular_season_df['Season'] == 2025]
active_w_teams_ids = pd.concat([womens_2025['WTeamID'], womens_2025['LTeamID']]).unique()
active_w_teams = w_teams_df[w_teams_df['TeamID'].isin(active_w_teams_ids)]
print(len(active_m_teams))
print(len(active_w_teams))

# Generate matchups for the year 2025
year = 2025
m_matchups = []
w_matchups = []

for team1, team2 in combinations(active_m_teams['TeamID'], 2):
    lower_id = min(team1, team2)
    higher_id = max(team1, team2)
    m_matchups.append(f"{year}_{lower_id}_{higher_id}")

for team1, team2 in combinations(w_teams_df['TeamID'], 2):
    lower_id = min(team1, team2)
    higher_id = max(team1, team2)
    w_matchups.append(f"{year}_{lower_id}_{higher_id}")

# Create DataFrames for the submissions
m_submission_df = pd.DataFrame({
    'ID': m_matchups,
    'Pred': [0.5] * len(m_matchups)  # Set all probabilities to 0.5
})

w_submission_df = pd.DataFrame({
    'ID': w_matchups,
    'Pred': [0.5] * len(w_matchups)  # Set all probabilities to 0.5
})

# Combine the men's and women's submissions
combined_submission_df = pd.concat([m_submission_df, w_submission_df], ignore_index=True)

# Save the combined submission DataFrame to a CSV file
submission_path = '../data/submission/submission.csv'
combined_submission_df.to_csv(submission_path, index=False)

print(f"Submission file successfully created at {submission_path}")