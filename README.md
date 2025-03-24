# 🏀 March Madness Game Prediction Model
This project predicts the outcomes of NCAA March Madness tournament games using machine learning models. It leverages features such as regular season predictions, team seeds, and other game-related statistics to train and evaluate models, with a focus on optimizing Brier Score for probabilistic predictions. Predictions are done for both Mens and Womens march madness games.

This project is a part of the March ML Mania 2025 Kaggle competition where participants compete to seee who can have the most accurate predictions for march madness games, evaulated based on their Brier score. 

## Data 
This project leverages a variety of historical statistics from both Mens and Womens College Basketball. Most of the data was provided from Kaggle, while certain advanced statistics were pulled from external sources to provide a more robust ML solution. 

**Additional Sources:** 
 - [Sports Reference](https://SportsReference.com): Used this resource to pull various advanced statistics for Mens basketball teams, including SRS, SOS, Pace, 3PAr and FTr 
 - [College Basketball Data](https://CollegeDasketballData.com): Leveraged their API to pull historical offensive and defensive basketball rankings for mens teams. Also used the API to find the average height for each mens basketball team going back to 2014 

 ## Model Pipeline
The project follows these steps:
1. **Data Preprocessing:** Clean and merge regular season, tournament, and team data. 
2. **Feature Engineering:** Create statistical features such as team efficiency, seeding impact, and historical performance.
3. **Model Training:** Train models (e.g., Logistic Regression, XGBoost) with hyperparameter tuning.
4. **Evaluation:** Optimize and validate models using cross-validation and Brier score.
5. **Prediction:** Generate probabilistic predictions for tournament outcomes.

## Contact
For questions or suggestions, contact:
- GitHub: [alexreid-14](https://github.com/alexreid-14)
- Email: alexwreid144@gmail.com
