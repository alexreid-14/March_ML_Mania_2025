# 🏀 March Madness Game Prediction Model
This project predicts the outcomes of NCAA March Madness tournament games using machine learning models. It leverages features such as regular season predictions, team seeds, and other game-related statistics to train and evaluate models, with a focus on optimizing Brier Score for probabilistic predictions. Predictions are done for both Mens and Womens march madness games.

## Results
This project is a part of the March ML Mania 2025 Kaggle competition where participants compete to seee who can have the most accurate predictions for march madness games, evaulated based on their Brier score. 

**[Outcome](https://www.kaggle.com/certification/competitions/alexreid16/march-machine-learning-mania-2025):** Finished in the top **94th** percentile (97th out of 1727) with a Brier score of **.11063**

## Data 
This project leverages a variety of historical statistics from both Mens and Womens College Basketball. Most of the data was provided from Kaggle, while certain advanced statistics were pulled from external sources to provide a more robust ML solution. 

**Additional Sources:** 
 - [Sports Reference](https://SportsReference.com): Used this resource to pull various advanced statistics for Mens basketball teams, including SRS, SOS, Pace, 3PAr and FTr 
 - [College Basketball Data](https://collegebasketballdata.com): Leveraged their API to pull historical offensive and defensive basketball rankings for mens teams. Also used the API to find the average height for each mens basketball team going back to 2014 

 ## Model Pipeline
The project follows these steps:
1. **Preprocessing**
     - **Men's:** Aggregated men's team statistics over each season, pulled advanced stats from APIs and external data sources, saved results as a CSV.
     - **Women's:** Aggregated women's team statistics over each season, saved results as a CSV.
     - **ML:** Prepared the preprocessed data to be ready for the ML model. Included combining men's season summary statistics from a variety of sources and adding team summary statistics to historical game outcomes so it could be used for modeling.

2. **Regular Season Model Training (Mens and Womens):** 
Using the data created in the preprocessing steps, I evaulated various ML models to identify which was the most effective at predicting the outcomes of regular season games. After evaluating the importance and correlation of various features, I selected the optimal feature subset to be incldued in my ML model. I ended up using XGBoost to create the model and included the predicted outcome of this model as a feature to be used in March Madness predictions. I used this approach because there was a much larger quantity of regular season data to be used for training the ML model which allowed me to not overfit to the small sample size of post season games. 

3. **Post Season Model Training (Mens and Womens):** After I incorporated the regular season prediction into the feature set, I began training a model to predict post season outcomes. The features that I knew must be included were the regular season prediction I made earlier as well as a one-hot encoded feature representing the seed of a team in the tournament. For the rest of the features, derived from team season summary statistics, I created a [`program`](modeling/final_model.py) that selected a random subset of features from my features list and evaluated their performance based on their Brier score (the objective of the competition). I also used **RandomizedSearchCV** to find the optimized set of hyperparameters for the model. Once this was completed. I evaluated how the model performed with each distinct season as a test set. 

4. **Submission:** Once I was confident in the model and features that I chose, it was time to create the submission file. The requirements of the submission file were to be in the following format: `<year>_<team1>_<team2>, pred`, with the Team1 ID always being smaller than the Team2 ID to avoid any duplicates. To achieve this, I created all possible combinations of Team1 and Team2 with Team1 always having the lower ID. Then, I loaded my chosen model in and added the corresponding features used in the model to each team to have it make a prediction. I needed to predict every combination of Mens and Womens teams regardless if they were in the playoffs, so to do this I just set the prediction to 0.5 for all teams that did not have a seed (They did not make the NCAA Tournament). I also changed the prediction to 1 or 0 for all games where the seed difference between the two teams was greater than 10. This is because these upsets are unlikely to happen and giving absolute predictions beneifts my brier score. Lastly I outputed the predictions to a csv file so it could be submitted to the competition. 

## Features Used in ML Models

The following features are used in the machine learning models, along with their definitions in basketball terms:

- **Points Per Game:** The average number of points scored by a team per game during the season.
- **FG Percentage:** The field goal percentage of a team, calculated as the ratio of successful field goals made to the total attempted.
- **Threes Per Game:** The average number of three-point shots made per game by a team.
- **Three Point Percentage:** The percentage of successful three-point shots made by a team, calculated as the ratio of made three-pointers to the total attempted.
- **Free Throws Per Game:** The average number of free throws made per game by a team.
- **Free Throw Percentage:** The percentage of successful free throws made by a team, calculated as the ratio of free throws made to free throws attempted.
- **Offensive Rebound Rate:** The percentage of available offensive rebounds secured by a team, calculated as offensive rebounds divided by the sum of offensive rebounds and opponent’s defensive rebounds.
- **Defensive Rebound Rate:** The percentage of available defensive rebounds secured by a team, calculated as defensive rebounds divided by the sum of defensive rebounds and opponent’s offensive rebounds.
- **Turnovers Per Game:** The average number of turnovers committed per game by a team.
- **Opp FG Percentage:** The opponent's field goal percentage against a team, indicating how well opposing teams shoot against them.
- **Opp Three Point Percentage:** The opponent’s three-point shooting percentage against a team.
- **Opp Free Throws Per Game:** The average number of free throws made by opponents per game against a team.
- **Opp Turnovers Per Game:** The average number of turnovers forced per game by a team from their opponents.
- **Opp Threes Per Game:** The average number of three-point shots made by opponents per game against a team.
- **Turnover Margin:** The difference between turnovers committed by a team and turnovers forced on their opponents. A positive margin means a team forces more turnovers than they commit.
- **Win pct last 10 games:** The winning percentage of a team over their last 10 games.
- **SRS (Simple Rating System):** A rating that accounts for margin of victory and strength of schedule. Positive values indicate a stronger team.
- **SOS (Strength of Schedule):** The strength of a team’s schedule, considering the strength of opponents faced.
- **FTr (Free Throw Rate):** The ratio of free throw attempts to field goal attempts for a team, reflecting how often they get to the free-throw line.
- **Offensive Rating:** An estimate of points scored by a team per 100 possessions.
- **Defensive Rating:** An estimate of points allowed by a team per 100 possessions.
- **Pace:** The average number of possessions per game for a team, indicating their tempo of play.
- **3PAr (Three-Point Attempt Rate):** The ratio of three-point attempts to total field goal attempts by a team, indicating their reliance on three-point shooting.
- **Avg Height:** The average height of players on a team, potentially indicating size advantage.

## Files in This Repository

- [`data/`](data/): Contains all the data files used in this project
- [`preprocessing/`](preprocessing/): Contains all the files used to clean the data and create summary statsitics for each season
- [`modeling/`](modeling/): Contains Python scripts for regular season and post season model training
- [`submission/`](submission/): Contains the notebook used to generate the submission file

## Contact
For questions or suggestions, contact:
- GitHub: [alexreid-14](https://github.com/alexreid-14)
- Email: alexwreid144@gmail.com
