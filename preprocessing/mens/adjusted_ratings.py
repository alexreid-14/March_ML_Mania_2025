import os 
import requests 
import pandas as pd 
from dotenv import load_dotenv
from pathlib import Path 

load_dotenv() 
api_key = os.getenv('API_KEY')

base_url = "https://api.collegebasketballdata.com"


headers = {
    "Authorization":f"Bearer {api_key}"
}


years = list(range(2014, 2026))

def get_adjusted_ratings(years):
    team_stats = [] 
    for year in years: 
        url = f"{base_url}/ratings/adjusted?season={year}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200: 
             season_data = response.json()
             for teams in season_data: 
                 row = {
                     "season": teams['season'],
                     'teamID': teams['teamId'],
                     'team': teams['team'],
                     'offensiveRating': teams['offensiveRating'],
                     'defensiveRating': teams['defensiveRating'],
                     'netRating': teams['netRating']
                 }
                 team_stats.append(row)
                 
        else:
            print(f"Failed to retrieve data {response.status_code}") 
    df = pd.DataFrame(team_stats)
    return df 

def get_srs_ratings(years):
    team_stats = [] 
    for year in years: 
        url = f"{base_url}/ratings/srs?season={year}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200: 
             season_data = response.json()
             for teams in season_data: 
                 row = {
                     "season": teams['season'],
                     'teamID': teams['teamId'],
                     'team': teams['team'], 
                     'srs_rating': teams['rating']
                 }
                 team_stats.append(row)
                 
        else:
            print(f"Failed to retrieve data {response.status_code}") 
    df = pd.DataFrame(team_stats)
    return df 

def get_team_height(years):
    team_stats = []  # Store team data

    for year in years:
        url = f"{base_url}/teams/roster?season={year}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            season_data = response.json()

            # Loop through each team in the season data
            for team in season_data:
                # Extract player heights from the players list
                player_heights = [player['height'] for player in team['players'] if player['height'] is not None]

                # Calculate average height if there are players
                avg_height = sum(player_heights) / len(player_heights) if player_heights else None

                # Add data to row
                row = {
                    "season": team['season'],
                    "teamID": team['teamId'],
                    "avg_height": round(avg_height, 2) if avg_height else None,  # Round to 2 decimals
                }
                team_stats.append(row)

        else:
            print(f"Failed to retrieve data for {year}. Status code: {response.status_code}")
    # Create and return the DataFrame
    df = pd.DataFrame(team_stats)
    return df

adjusted_ratings =  get_adjusted_ratings(years) 
new_row = pd.DataFrame([{'season': 2017, 'teamID': 80, 'team':'Eastern Washington', 'offensiveRating': 111.3, 'defensiveRating': 113.6, 'netRating':-2.3 }])
adjusted_ratings = pd.concat([adjusted_ratings, new_row], ignore_index=True)

average_height = get_team_height(years) 

adjusted_ratings = adjusted_ratings.merge(average_height, how='left', on=['season', 'teamID'])


output_dir = Path("../..") / "data" / "preprocessing"
output_dir.mkdir(parents=True, exist_ok =True)
output_path = output_dir / "mens_season_ratings.csv"
adjusted_ratings.to_csv(output_path, index=False)
print(f"File successfully exported to {output_path}")
