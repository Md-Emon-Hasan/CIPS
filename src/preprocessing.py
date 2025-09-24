# preprocessing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def calculate_inning_runs(delivery_df):
    """Calculates the total runs scored in each inning of each match."""
    try:
        logging.info("Calculating total runs scored in each inning.")
        total_score_df = delivery_df.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
        logging.info(f"Shape of total score dataframe: {total_score_df.shape}")
        return total_score_df
    except Exception as e:
        logging.error(f"Error calculating inning runs: {e}")
        raise

def filter_first_innings(total_score_df):
    """Filters the dataframe to include only first innings scores."""
    try:
        logging.info("Filtering for first innings scores.")
        first_innings_df = total_score_df[total_score_df['inning'] == 1].copy()
        logging.info(f"Shape of first innings dataframe: {first_innings_df.shape}")
        return first_innings_df
    except Exception as e:
        logging.error(f"Error filtering first innings: {e}")
        raise

def visualize_total_runs_distribution(first_innings_df):
    """Creates a histogram to visualize the distribution of total runs in first innings."""
    try:
        logging.info("Visualizing distribution of total runs in first innings.")
        plt.figure(figsize=(8, 5))
        sns.histplot(first_innings_df['total_runs'], bins=30, kde=False)
        plt.title('Distribution of Total Runs')
        plt.xlabel('Total Runs')
        plt.ylabel('Number of Match')
        plt.show()
        logging.info("Visualization of total runs distribution completed.")
    except Exception as e:
        logging.error(f"Error visualizing total runs distribution: {e}")
        raise

def merge_total_runs_with_match(match_df, total_score_df):
    """Merges the total runs data with the match dataframe."""
    try:
        logging.info("Merging total runs with match dataframe.")
        match_runs_df = match_df.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on="match_id")
        logging.info(f"Shape of merged match dataframe: {match_runs_df.shape}")
        return match_runs_df
    except Exception as e:
        logging.error(f"Error merging total runs with match: {e}")
        raise

def filter_teams(match_runs_df, teams):
    """Filters the dataframe to include only matches between the specified teams."""
    try:
        logging.info(f"Filtering matches for teams: {teams}")
        filtered_df = match_runs_df[match_runs_df['team1'].isin(teams)].copy()
        filtered_df = filtered_df[filtered_df['team2'].isin(teams)].copy()
        logging.info(f"Shape of filtered dataframe: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logging.error(f"Error filtering teams: {e}")
        raise

def standardize_team_names(match_df):
    """Standardizes the team names in the dataframe."""
    try:
        logging.info("Standardizing team names.")
        match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
        match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
        match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
        match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
        logging.info("Team names standardized.")
        return match_df
    except Exception as e:
        logging.error(f"Error standardizing team names: {e}")
        raise

def filter_dl_applied(match_df):
    """Filters out matches where Duckworth-Lewis method was applied."""
    try:
        logging.info("Filtering out matches with DL applied.")
        filtered_df = match_df[match_df['dl_applied'] == 0].copy()
        logging.info(f"Shape after filtering DL applied matches: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logging.error(f"Error filtering DL applied matches: {e}")
        raise

def select_relevant_match_columns(match_df):
    """Selects only the relevant columns from the match dataframe."""
    try:
        logging.info("Selecting relevant columns from match dataframe.")
        relevant_df = match_df[['match_id', 'city', 'winner', 'total_runs']].copy()
        logging.info(f"Shape after selecting relevant match columns: {relevant_df.shape}")
        return relevant_df
    except Exception as e:
        logging.error(f"Error selecting relevant match columns: {e}")
        raise

def merge_match_delivery(match_df, delivery_df):
    """Merges the match dataframe with the delivery dataframe."""
    try:
        logging.info("Merging match and delivery dataframes.")
        merged_df = match_df.merge(delivery_df, on='match_id')
        logging.info(f"Shape of merged dataframe: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logging.error(f"Error merging match and delivery dataframes: {e}")
        raise

def filter_second_innings(merged_df):
    """Filters the dataframe to include only second innings data."""
    try:
        logging.info("Filtering for second innings data.")
        second_innings_df = merged_df[merged_df['inning'] == 2].copy()
        logging.info(f"Shape after filtering for second innings: {second_innings_df.shape}")
        return second_innings_df
    except Exception as e:
        logging.error(f"Error filtering second innings: {e}")
        raise

def calculate_cumulative_runs(delivery_df):
    """Calculates cumulative runs scored over the course of each match."""
    try:
        logging.info("Calculating cumulative runs.")
        delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
        logging.info("Cumulative runs calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating cumulative runs: {e}")
        raise

def calculate_runs_left(delivery_df):
    """Calculates runs remaining to win the match."""
    try:
        logging.info("Calculating runs left.")
        delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score'] + 1
        logging.info("Runs left calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating runs left: {e}")
        raise

def calculate_balls_left(delivery_df):
    """Calculates balls remaining in the innings."""
    try:
        logging.info("Calculating balls left.")
        delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
        logging.info("Balls left calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating balls left: {e}")
        raise

def handle_player_dismissals(delivery_df):
    """Handles player dismissals by filling NA values with 0 and converting to binary."""
    try:
        logging.info("Handling player dismissals.")
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 1 if x != 0 else 0)
        logging.info("Player dismissals handled.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error handling player dismissals: {e}")
        raise

def calculate_wickets_left(delivery_df):
    """Calculates wickets remaining in the innings."""
    try:
        logging.info("Calculating wickets left.")
        wickets_fallen = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
        delivery_df['wickets'] = 10 - wickets_fallen
        logging.info("Wickets left calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating wickets left: {e}")
        raise

def calculate_current_run_rate(delivery_df):
    """Calculates the current run rate."""
    try:
        logging.info("Calculating current run rate.")
        delivery_df['crr'] = delivery_df.current_score * 6 / (120 - delivery_df.balls_left)
        logging.info("Current run rate calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating current run rate: {e}")
        raise

def calculate_required_run_rate(delivery_df):
    """Calculates the required run rate."""
    try:
        logging.info("Calculating required run rate.")
        delivery_df['rrr'] = delivery_df.runs_left * 6 / delivery_df.balls_left
        logging.info("Required run rate calculated.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error calculating required run rate: {e}")
        raise

def reset_dataframe_index(delivery_df):
    """Resets the dataframe index and drops the old index column."""
    try:
        logging.info("Resetting dataframe index.")
        delivery_df = delivery_df.reset_index(drop=True)
        logging.info("Index reset.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error resetting dataframe index: {e}")
        raise

def create_winner_column(delivery_df):
    """Creates a new column indicating whether the batting team won."""
    try:
        logging.info("Creating winner column.")
        winner = []
        for item, row in delivery_df.iterrows():
            if row.winner == row.batting_team:
                winner.append(1)
            else:
                winner.append(0)
        delivery_df['winner'] = winner
        logging.info("Winner column created.")
        return delivery_df
    except Exception as e:
        logging.error(f"Error creating winner column: {e}")
        raise

def select_final_columns(delivery_df):
    """Selects the final set of columns for the model."""
    try:
        logging.info("Selecting final columns for the model.")
        final_df = delivery_df[['match_id', 'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'winner']].copy()
        logging.info(f"Shape of final dataframe before shuffling: {final_df.shape}")
        return final_df
    except Exception as e:
        logging.error(f"Error selecting final columns: {e}")
        raise

def shuffle_dataframe(final_df):
    """Shuffles the dataframe to randomize the order of samples."""
    try:
        logging.info("Shuffling the dataframe.")
        shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info("Dataframe shuffled.")
        return shuffled_df
    except Exception as e:
        logging.error(f"Error shuffling dataframe: {e}")
        raise

def standardize_final_team_names(final_df):
    """Standardizes team names in the final dataframe."""
    try:
        logging.info("Standardizing team names in final dataframe.")
        final_df['batting_team'] = final_df['batting_team'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
        final_df['bowling_team'] = final_df['bowling_team'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
        final_df['batting_team'] = final_df['batting_team'].str.replace('Delhi Daredevils', 'Delhi Capitals')
        final_df['bowling_team'] = final_df['bowling_team'].str.replace('Delhi Daredevils', 'Delhi Capitals')
        logging.info("Team names standardized in final dataframe.")
        return final_df
    except Exception as e:
        logging.error(f"Error standardizing final team names: {e}")
        raise

def handle_missing_city(final_df, cities_dict):
    """Handles missing city values by mapping teams to their home cities."""
    try:
        logging.info("Handling missing city values.")
        final_df['city'] = final_df['city'].fillna(0)

        def fill_city(x):
            if x.city == 0:
                team = [x.batting_team, x.bowling_team][random.randint(0, 1)]
                return cities_dict[team]
            else:
                return x.city

        final_df['city'] = final_df.apply(fill_city, axis=1)
        logging.info("Missing city values handled.")
        return final_df
    except Exception as e:
        logging.error(f"Error handling missing city values: {e}")
        raise

def drop_na_values(final_df):
    """Drops any remaining NA values from the dataframe."""
    try:
        logging.info("Dropping NA values.")
        final_df.dropna(inplace=True)
        logging.info(f"Shape after dropping NA values: {final_df.shape}")
        return final_df
    except Exception as e:
        logging.error(f"Error dropping NA values: {e}")
        raise

def remove_zero_balls_left(final_df):
    """Removes rows where balls_left is 0."""
    try:
        logging.info("Removing rows where balls_left is 0.")
        final_df = final_df[final_df.balls_left != 0].copy()
        logging.info(f"Shape after removing rows with 0 balls left: {final_df.shape}")
        return final_df
    except Exception as e:
        logging.error(f"Error removing zero balls left: {e}")
        raise

def drop_match_id(final_df):
    """Drops the match_id column from the final dataframe."""
    try:
        logging.info("Dropping match_id column.")
        final_df.drop(columns=['match_id'], inplace=True)
        logging.info("match_id column dropped.")
        return final_df
    except Exception as e:
        logging.error(f"Error dropping match_id column: {e}")
        raise

def split_data(final_df, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    try:
        logging.info(f"Splitting data into training and testing sets with test_size={test_size} and random_state={random_state}.")
        from sklearn.model_selection import train_test_split
        X = final_df.drop(columns=['winner'])
        y = final_df['winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

if __name__ == '__main__':
    from data_collection import load_data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    data_dir = os.path.join(BASE_DIR, 'data')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    matches_path = os.path.join(data_dir, 'matches.csv')

    # Load data using the paths constructed dynamically
    delivery_data, match_data = load_data(deliveries_path, matches_path)

    total_score_df = calculate_inning_runs(delivery_data.copy())
    first_innings_df = filter_first_innings(total_score_df.copy())
    visualize_total_runs_distribution(first_innings_df.copy())
    match_runs_df = merge_total_runs_with_match(match_data.copy(), total_score_df.copy())

    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab',
        'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals'
    ]
    filtered_match_df = filter_teams(match_runs_df.copy(), teams)
    standardized_match_df = standardize_team_names(filtered_match_df.copy())
    dl_filtered_match_df = filter_dl_applied(standardized_match_df.copy())
    relevant_match_df = select_relevant_match_columns(dl_filtered_match_df.copy())
    delivery_df = merge_match_delivery(relevant_match_df.copy(), delivery_data.copy())
    second_innings_delivery_df = filter_second_innings(delivery_df.copy())
    cumulative_runs_df = calculate_cumulative_runs(second_innings_delivery_df.copy())
    runs_left_df = calculate_runs_left(cumulative_runs_df.copy())
    balls_left_df = calculate_balls_left(runs_left_df.copy())
    dismissals_handled_df = handle_player_dismissals(balls_left_df.copy())
    wickets_left_df = calculate_wickets_left(dismissals_handled_df.copy())
    crr_df = calculate_current_run_rate(wickets_left_df.copy())
    rrr_df = calculate_required_run_rate(crr_df.copy())
    reset_index_df = reset_dataframe_index(rrr_df.copy())
    winner_added_df = create_winner_column(reset_index_df.copy())
    final_df = select_final_columns(winner_added_df.copy())
    shuffled_final_df = shuffle_dataframe(final_df.copy())
    final_df_standardized_teams = standardize_final_team_names(shuffled_final_df.copy())

    cities_dict = {
        'Royal Challengers Bangalore': 'Bengaluru',
        'Chennai Super Kings': 'Chennai',
        'Kings XI Punjab': 'Mumbai',
        'Kolkata Knight Riders': 'Kolkata',
        'Delhi Capitals': 'Delhi',
        'Rajasthan Royals': 'Jaipur',
        'Mumbai Indians': 'Mumbai',
        'Sunrisers Hyderabad': 'Hyderabad',
    }
    final_df_handled_city = handle_missing_city(final_df_standardized_teams.copy(), cities_dict)
    final_df_no_na = drop_na_values(final_df_handled_city.copy())
    final_df_valid_balls = remove_zero_balls_left(final_df_no_na.copy())
    final_df_no_match_id = drop_match_id(final_df_valid_balls.copy())

    X_train, X_test, y_train, y_test = split_data(final_df_no_match_id.copy())

    print("\nFirst few rows of processed data for model training:")
    print(X_train.head())
    print("\nShape of training data:", X_train.shape)
    print("Shape of testing data:", X_test.shape)