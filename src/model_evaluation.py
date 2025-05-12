# model_evaluation.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, max_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluates the given model on the test set and prints the evaluation metrics."""
    try:
        logging.info(f"Evaluating {model_name} model.")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        median_ae = median_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)

        print(f"----- {model_name} Model Evaluation Metrics -----")
        print(f"R2 Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Median Absolute Error: {median_ae:.4f}")
        print(f"Max Error: {max_err:.4f}")
        print(f"Explained Variance Score: {explained_variance:.4f}")
        logging.info(f"{model_name} model evaluation completed.")
        return r2, mae, mse, rmse, median_ae, max_err, explained_variance
    except Exception as e:
        logging.error(f"Error evaluating {model_name} model: {e}")
        raise

def evaluate_classification_model(model_name, model, X_test, y_test):
    """Evaluates the given classification model on the test set and prints metrics."""
    try:
        logging.info(f"Evaluating {model_name} classification model.")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        median_ae = median_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"----- {model_name} Classification Model Evaluation Metrics -----")
        print(f"R2 Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Median Absolute Error: {median_ae:.4f}")
        print(f"Max Error: {max_err:.4f}")
        print(f"Explained Variance Score: {explained_variance:.4f}")
        print(f"Accuracy Score: {accuracy:.4f}")
        logging.info(f"{model_name} classification model evaluation completed.")
        return r2, mae, mse, rmse, median_ae, max_err, explained_variance, accuracy
    except Exception as e:
        logging.error(f"Error evaluating {model_name} classification model: {e}")
        raise

def train_and_evaluate(model_name, model, X_train, X_test, y_test, save_path=None):
    """Trains the model, evaluates it, and optionally saves it."""
    try:
        logging.info(f"Training {model_name} model.")
        step1 = ColumnTransformer(transformers=[
            ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
        ], remainder='passthrough')
        pipe = Pipeline([
            ('step1', step1),
            ('step2', model)
        ])
        pipe.fit(X_train, y_train)
        logging.info(f"{model_name} model trained successfully.")

        if isinstance(model, LogisticRegression):
            evaluate_classification_model(model_name, pipe, X_test, y_test)
        else:
            evaluate_model(model_name, pipe, X_test, y_test)

        if save_path:
            try:
                logging.info(f"Saving {model_name} model to: {save_path}")
                pickle.dump(pipe, open(save_path, 'wb'))
                logging.info(f"{model_name} model saved successfully.")
            except Exception as e:
                logging.error(f"Error saving {model_name} model: {e}")

        return pipe
    except Exception as e:
        logging.error(f"Error training and evaluating {model_name} model: {e}")
        raise

if __name__ == '__main__':
    from preprocessing import split_data
    from data_collection import load_data

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder where this script is located
    data_dir = os.path.join(BASE_DIR, 'data')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    matches_path = os.path.join(data_dir, 'matches.csv')

    # Load data using the paths constructed dynamically
    delivery_data, match_data = load_data(deliveries_path, matches_path)

    total_score_df = pd.read_csv('temp_total_score.csv') # Assuming you saved this during preprocessing
    match_runs_df = match_data.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on="match_id")
    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab',
        'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals'
    ]
    match_df = match_runs_df[match_runs_df['team1'].isin(teams) & match_runs_df['team2'].isin(teams)].copy()
    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad').str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad').str.replace('Delhi Daredevils', 'Delhi Capitals')
    final_df = pd.read_csv('temp_final_df.csv') # Assuming you saved this during preprocessing

    X_train, X_test, y_train, y_test = split_data(final_df.drop(columns=['match_id']).copy())

    train_and_evaluate('Linear Regression', LinearRegression(), X_train, X_test, y_train, save_path='../model/linear_regression_pipe.pkl')
    train_and_evaluate('Ridge Regression', Ridge(alpha=10), X_train, X_test, y_train, save_path='../model/ridge_regression_pipe.pkl')
    train_and_evaluate('Lasso Regression', Lasso(alpha=0.001), X_train, X_test, y_train, save_path='../model/lasso_regression_pipe.pkl')
    train_and_evaluate('KNN', KNeighborsRegressor(n_neighbors=3), X_train, X_test, y_train, save_path='../model/knn_pipe.pkl')
    train_and_evaluate('Decision Tree', DecisionTreeRegressor(max_depth=8), X_train, X_test, y_train, save_path='../model/decision_tree_pipe.pkl')
    train_and_evaluate('Random Forest', RandomForestRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15), X_train, X_test, y_train, save_path='../model/random_forest_pipe.pkl')
    train_and_evaluate('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=3, max_features=0.75, max_depth=15), X_train, X_test, y_train, save_path='../model/extra_trees_pipe.pkl')
    train_and_evaluate('AdaBoost', AdaBoostRegressor(n_estimators=15, learning_rate=1.0), X_train, X_test, y_train, save_path='../model/adaboost_pipe.pkl')
    train_and_evaluate('Gradient Boost', GradientBoostingRegressor(n_estimators=15, learning_rate=1.0), X_train, X_test, y_train, save_path='../model/gradient_boost_pipe.pkl')
    train_and_evaluate('XGBoost', XGBRegressor(n_estimators=15, learning_rate=1.0), X_train, X_test, y_train, save_path='../model/xgboost_pipe.pkl')
    train_and_evaluate('Logistic Regression', LogisticRegression(solver='liblinear'), X_train, X_test, y_train, save_path='../model/logistic_regression_pipe.pkl')