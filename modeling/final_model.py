import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from sklearn.metrics import brier_score_loss 
from sklearn.metrics import accuracy_score 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV



def train_and_save_logistic_regression(X, y, model_filename='log_reg_model.joblib', test_size=0.2, random_state=57):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Identify columns to scale (exclude seed-related columns)
    scaled_columns = [col for col in X.columns if 'seed' not in col.lower()]
    passthrough_columns = [col for col in X.columns if 'seed' in col.lower()]

    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), scaled_columns),  # Scale selected columns
            ('passthrough', 'passthrough', passthrough_columns)  # Leave seed columns unchanged
        ])
    
    # Transform the training and testing data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Train the Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(X_train_transformed, y_train)

    # Save the trained model
    joblib.dump(logreg, model_filename)
    print(f"Model saved as {model_filename}")
    return logreg


def train_and_tune_xgboost(X, y, model_filename="xgb_model.json", param_search=True, n_iter=20, cv=3, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state)
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
            verbose=2,
            n_jobs=-1,
            random_state=random_state
        )

        # Fit the model with hyperparameter tuning
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        print("Best Parameters:", best_params)

        # Use the best model
        xgb_model = random_search.best_estimator_
    else:
        print("Training with default hyperparameters...")
        xgb_model.fit(X_train, y_train)

    # Evaluate Brier Score on test set
    y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_probs)
    print(f"Brier Score on test set: {brier:.4f}")

    # Save the trained model
    xgb_model.save_model(model_filename)
    print(f"Model saved as {model_filename}")
    return xgb_model

def test_model_on_each_season(model, df, feature_columns):
    seasons = df['Season'].unique()
    brier_scores = {}

    for test_season in seasons:
        print(f"Testing model on Season {test_season}...")
        test_data = df[df['Season'] == test_season]
        X_test = test_data[feature_columns]
        y_test = test_data['Team1_Wins']

        # Predict probabilities
        y_pred_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Team1 winning)

        # Compute Brier Score
        brier = brier_score_loss(y_test, y_pred_probs)
        brier_scores[test_season] = brier
        print(f"Season {test_season} - Brier Score: {brier:.4f}")

    # Print summary
    print("\nBrier Scores by Season:")
    for season, score in brier_scores.items():
        print(f"Season {season}: {score:.4f}")

    # Average Brier Score across all seasons
    avg_brier = np.mean(list(brier_scores.values()))
    print(f"\nAverage Brier Score: {avg_brier:.4f}")
    return brier_scores, avg_brier



df = pd.read_csv("../data/modeling/final_ml.csv")
df = df[['Season', 'Team1_Wins', 'reg_season_pred', 'Seed_1', 'Seed_2']]
df = pd.get_dummies(df, columns=['Seed_1', 'Seed_2'], prefix=['T1_Seed','T2_Seed'], dtype=int)

features = df.drop(columns=['Season', 'Team1_Wins'])
feature_columns = df.drop(columns=['Season', 'Team1_Wins']).columns
target = df['Team1_Wins']

# Load XGB model 
xgb_model = train_and_tune_xgboost(features, target)

# Load LogReg model 
log_model = train_and_save_logistic_regression(features, target)
#log_model = joblib.load('log_reg_model.joblib')

# Evaluate the model 
brier_scores, avg_brier = test_model_on_each_season(xgb_model, df, feature_columns)
print('Done!')