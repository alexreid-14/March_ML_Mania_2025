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
                     'srs_rating': teams['rating'],
                 }
                 team_stats.append(row)
                 
        else:
            print(f"Failed to retrieve data {response.status_code}") 
    df = pd.DataFrame(team_stats)
    return df 



adjusted_ratings =  get_adjusted_ratings(years) 
srs_ratings = get_srs_ratings(years) 

merged_df = adjusted_ratings.merge(srs_ratings, how="left", on=['teamID', 'season'])

output_dir = Path("..") / "data" / "preprocessing"
output_dir.mkdir(parents=True, exist_ok =True)
output_path = output_dir / "mens_season_ratings.csv"
merged_df.to_csv(output_path, index=False)