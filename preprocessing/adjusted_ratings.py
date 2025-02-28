import os 
import requests 
import pandas as pd 
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv('API_KEY')

base_url = "https://api.collegebasketballdata.com"


headers = {
    "Authorization":f"Bearer {api_key}"
}

years = list(range(2014, 2025))

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
                     'name': teams['team'],
                     'offensiveRating': teams['offensiveRating'],
                     'defensiveRating': teams['defensiveRating'],
                     'netRating': teams['netRating']
                 }
                 team_stats.append(row)
                 
        else:
            print(f"Failed to retrieve data {response.status_code}") 
    df = pd.DataFrame(team_stats)
    print(df.head(5)) 


get_adjusted_ratings(years) 
