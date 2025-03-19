import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss 
from sklearn.metrics import accuracy_score 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def train_and_tune_xgboost(X, y, model_filename="xgb_model.json", param_search=True, n_iter=20, cv=5, random_state=49):
    # Define parameter grid for tuning
    param_dist = {
        'n_estimators': np.arange(100, 1001, 100),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'max_depth': np.arange(3, 10),
        'min_child_weight': np.arange(1, 10),
        'subsample': np.linspace(0.5, 1.0, 5),
        'colsample_bytree': np.linspace(0.5, 1.0, 5)
    }

    # Initialize base XGBoost model
    xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state)
    best_params = None  # Placeholder for best parameters

    if param_search:
        print("Performing hyperparameter tuning...")

        # Randomized Search for Hyperparameter Tuning
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_brier_score',  # Minimize Brier Score
            cv=cv,
            verbose=0,
            n_jobs=-1,
            random_state=random_state
        )

        # Fit the model with hyperparameter tuning
        random_search.fit(X, y)  # Use the entire dataset
        best_params = random_search.best_params_
        print("Best Parameters:", best_params)

        # Use the best model
        xgb_model = random_search.best_estimator_
    else:
        print("Training with default hyperparameters...")
        xgb_model.fit(X, y)  # Use the entire dataset

    # Cross-validated Brier Score on the entire dataset
    cv_brier_scores = -cross_val_score(xgb_model, X, y, scoring='neg_brier_score', cv=cv)
    brier_score = np.mean(cv_brier_scores)
    print(f"Cross-validated Brier Score: {np.mean(cv_brier_scores):.4f} (±{np.std(cv_brier_scores):.4f})")

    # Save the trained model
    xgb_model.save_model(model_filename)
    print(f"Model saved to {model_filename}")

    return xgb_model, brier_score

def test_model_on_each_season(model, df, feature_columns):
    seasons = df['Season'].unique()
    brier_scores = []

    for test_season in seasons:
        test_data = df[df['Season'] == test_season]
        X_test = test_data[feature_columns]
        y_test = test_data['Team1_Wins']

        # Predict probabilities
        y_pred_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Team1 winning)

        # Compute Brier Score
        brier = float(brier_score_loss(y_test, y_pred_probs))
        
        brier_scores.append({
            'season': int(test_season),  # Convert season to int for serialization
            'brier_score': brier
        })

    # Print summary
    print("\nBrier Scores by Season:")
    for entry in brier_scores:
        print(f"Season {entry['season']}: {entry['brier_score']:.4f}")

    # Average Brier Score across all seasons
    avg_brier = np.mean([entry['brier_score'] for entry in brier_scores])
    return brier_scores, avg_brier


# Function to randomly select feature pairs
def select_feature_pairs(feature_pairs, num_pairs):
    indices = np.arange(len(feature_pairs))
    selected_indices = np.random.choice(indices, size=num_pairs, replace=False)
    selected_pairs = [feature_pairs[i] for i in selected_indices]
    return selected_pairs

# Function to convert non-serializable types to serializable ones
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):  # Handle NaN values
        return None
    else:
        return obj




mens_df = pd.read_csv("../data/modeling/final_ml.csv")
womens_df = pd.read_csv("../data/modeling/")

mens_feature_pairs = [
    #('FG_Percentage_1', 'FG_Percentage_2'),
    #('Defensive_Rebound_Rate_1', 'Defensive_Rebound_Rate_2'),
    #('Opp_FG_Percentage_1', 'Opp_FG_Percentage_2'),
    #('SOS_1', 'SOS_2'),
    ('Win_Percentage_1', 'Win_Percentage_2'),
    ('Three_Point_Percentage_1', 'Three_Point_Percentage_2'),
    ('Offensive_Rebound_Rate_1', 'Offensive_Rebound_Rate_2'),
    ('Turnovers_Per_Game_1', 'Turnovers_Per_Game_2'),
    ('Opp_Free_Throws_Per_Game_1', 'Opp_Free_Throws_Per_Game_2'),
    ('Opp_Turnovers_Per_Game_1', 'Opp_Turnovers_Per_Game_2'),
    ('Win_pct_last_10_games_1', 'Win_pct_last_10_games_2'),
    ('SRS_1', 'SRS_2'),
    ('Pace_1', 'Pace_2'),
    ('FTr_1', 'FTr_2'),
    ('3PAr_1', '3PAr_2'),
    ('offensiveRating_1', 'offensiveRating_2'),
    ('defensiveRating_1', 'defensiveRating_2')
]

womens_feature_pairs = [
    ('Win_Percentage_1', 'Win_Percentage_2'),
    ('FG_Percentage_1', 'FG_Percentage_2'),
    ('Threes_Per_Game_1', 'Threes_Per_Game_2'),
    ('Free_Throws_Per_Game_1', 'Free_Throws_Per_Game_2'),
    ('Free_Throw_Percentage_1', 'Free_Throw_Percentage_2'),

    ('Defensive_Rebound_Rate_1', 'Defensive_Rebound_Rate_2'),
    ('Opp_FG_Percentage_1', 'Opp_FG_Percentage_2'),
    ('Win_Percentage_1', 'Win_Percentage_2'),
    ('Three_Point_Percentage_1', 'Three_Point_Percentage_2'),
    ('Offensive_Rebound_Rate_1', 'Offensive_Rebound_Rate_2'),
    ('Turnovers_Per_Game_1', 'Turnovers_Per_Game_2'),
    ('Opp_Free_Throws_Per_Game_1', 'Opp_Free_Throws_Per_Game_2'),
    ('Opp_Turnovers_Per_Game_1', 'Opp_Turnovers_Per_Game_2'),
    ('Win_pct_last_10_games_1', 'Win_pct_last_10_games_2'),
    ('offensiveRating_1', 'offensiveRating_2'),
    ('defensiveRating_1', 'defensiveRating_2')
]



# Main loop to test different feature combinations
num_pairs_to_select = 6  # Number of feature pairs to select
num_iterations = 12 # Number of different combinations to try
results = []

for i in range(num_iterations):
    selected_pairs = select_feature_pairs(mens_feature_pairs, num_pairs_to_select)
    selected_features = [feature for pair in selected_pairs for feature in pair]
    print(f"Iteration {i+1}")

    df_subset = df[['Season', 'Team1_Wins', 'reg_season_pred', 'Seed_1', 'Seed_2'] + selected_features]
    df_subset = pd.get_dummies(df_subset, columns=['Seed_1', 'Seed_2'], prefix=['T1_Seed','T2_Seed'], dtype=int)

    features = df_subset.drop(columns=['Season', 'Team1_Wins'])
    feature_columns = df_subset.drop(columns=['Season', 'Team1_Wins']).columns
    target = df_subset['Team1_Wins']

    # Load XGB model 
    xgb_model, brier = train_and_tune_xgboost(features, target)

    # Evaluate the model 
    brier_scores, avg_brier = test_model_on_each_season(xgb_model, df_subset, feature_columns)

    results.append({
        'iteration': i + 1,
        'selected_features': selected_features,
        'model': xgb_model,
        'brier_score': brier,
        'avg_brier_score': avg_brier, 
        'brier_scores': brier_scores
    })

    print(f"Average Brier Score: {avg_brier}")
    print('-------------------------------')

# Find the best-performing feature subset
best_result = min(results, key=lambda x: x['brier_score'])

# Save the best model 
model_filename = "best_xgb_model.model"
best_result['model'].save_model(model_filename)
print(f"Model saved as {model_filename}")

# Save the best features as JSON
best_features_filename = "best_features.json"
with open(best_features_filename, "w") as f:
    json.dump({
        "selected_features": best_result['selected_features'],
        "brier_scores": best_result['brier_scores'],
        "brier_score": best_result['brier_score'],
        "avg_brier_score": best_result['avg_brier_score'],
        "model_parameters": convert_to_serializable(best_result['model'].get_params())
    }, f, indent=4)
print(f"Best features saved as {best_features_filename}")

# Print the best result
print("\nBest Feature Subset:")
print(f"Iteration: {best_result['iteration']}\n")
print(f"Selected Features: {best_result['selected_features']}\n")
print(f"Brier Score on Test Set: {best_result['brier_score']}\n")
print(f"Average Brier Score: {best_result['avg_brier_score']}\n")
print(f"Model Parameters: {best_result['model'].get_params()}\n")
print('Done!')














