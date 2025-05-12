# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def analyze_matches_per_season(match_df):
    """Analyzes and visualizes the number of matches played each season."""
    try:
        logging.info("Analyzing number of matches each season.")
        plt.figure()
        sns.countplot(x='Season', data=match_df)
        plt.title("Number of Matches Each Season")
        plt.xlabel("Season")
        plt.ylabel("Number of Matches")
        plt.show()
        logging.info("Number of matches each season analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing matches per season: {e}")
        raise

def analyze_teams_per_season(match_df):
    """Analyzes and visualizes the number of teams participated each season."""
    try:
        logging.info("Analyzing number of teams participating each season.")
        plt.figure(figsize=(15, 5))
        match_df.groupby('Season')['team1'].nunique().plot(kind='bar')
        plt.title("Number of teams participated each season", fontsize=18, fontweight="bold")
        plt.ylabel("Count of teams", size=25)
        plt.xlabel("Season", size=25)
        plt.xticks(size=15, rotation=0)
        plt.yticks(size=15)
        plt.show()
        logging.info("Number of teams participating each season analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing teams per season: {e}")
        raise

def analyze_matches_by_team1(match_df):
    """Analyzes and visualizes the count of matches played by each team in team1 column."""
    try:
        logging.info("Analyzing matches count by team1.")
        team1_counts = match_df['team1'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.countplot(x='team1', data=match_df, order=team1_counts.index)
        plt.xlabel('Team 1')
        plt.ylabel('Count')
        plt.title('Matches Count by Team 1')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        logging.info("Matches count by team1 analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing matches by team1: {e}")
        raise

def analyze_toss_winner(match_df):
    """Analyzes and visualizes the toss winners."""
    try:
        logging.info("Analyzing toss winners.")
        plt.figure()
        sns.countplot(x='toss_winner', data=match_df)
        plt.xticks(rotation='vertical')
        plt.title("Toss Winners")
        plt.xlabel("Toss Winner")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        logging.info("Toss winners analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing toss winners: {e}")
        raise

def analyze_matches_result(match_df):
    """Analyzes and visualizes the result of the matches."""
    try:
        logging.info("Analyzing match results.")
        result_counts = match_df['result'].value_counts()
        plt.figure(figsize=(8, 6))
        sns.countplot(x='result', data=match_df, order=result_counts.index)
        plt.xlabel('Result')
        plt.ylabel('Count')
        plt.title('Matches Result')
        plt.show()
        logging.info("Match results analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing match results: {e}")
        raise

def analyze_toss_match_winner(match_df):
    """Analyzes and visualizes if the toss winner also won the match."""
    try:
        logging.info("Analyzing if toss winner also won the match.")
        match_df['toss_match_winner'] = match_df['toss_winner'] == match_df['winner']
        plt.figure(figsize=(6, 4))
        sns.countplot(data=match_df, x='toss_match_winner', palette='Set2')
        plt.title('Did Toss Winner Also Win the Match?')
        plt.xlabel('Toss Winner == Match Winner')
        plt.ylabel('Match Count')
        plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
        plt.tight_layout()
        plt.show()
        logging.info("Toss match winner analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing toss match winner: {e}")
        raise

def analyze_winner_of_toss_each_season(match_df):
    """Analyzes and visualizes the number of teams who won after winning the toss each season."""
    try:
        logging.info("Analyzing teams who won after winning the toss each season.")
        winneroftoss = match_df[(match_df['toss_winner']) == (match_df['winner'])]
        plt.figure(figsize=(8, 6))
        wot = sns.countplot(x='winner', hue='Season', data=winneroftoss)
        plt.xticks(rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Teams")
        plt.ylabel("Number of Wins")
        plt.title("Number of Teams who won, given they win the toss, every season")
        plt.show(wot)
        logging.info("Teams who won after winning the toss each season analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing winner of toss each season: {e}")
        raise

def analyze_top_player_of_the_match(match_df, top_n=10):
    """Analyzes and visualizes the top N player of the match winners."""
    try:
        logging.info(f"Analyzing top {top_n} player of the match winners.")
        top_players = match_df.player_of_match.value_counts()[:top_n]
        fig, ax = plt.subplots()
        ax.set_ylim([0, top_n + 10])
        ax.set_ylabel("Number of Awards")
        ax.set_xlabel("Name of Players")
        ax.set_title("Top player of the match Winners")
        sns.barplot(x=top_players.index, y=top_players, orient='v', palette="RdBu")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.show()
        logging.info(f"Top {top_n} player of the match winners analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing top player of the match: {e}")
        raise

def analyze_team1_vs_winner(match_df):
    """Analyzes and visualizes the cross-tabulation between team1 and winner using a heatmap."""
    try:
        logging.info("Analyzing team1 vs winner cross-tabulation.")
        cross_tab = pd.crosstab(match_df['team1'], match_df['winner'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
        plt.xlabel('Winner')
        plt.ylabel('Team 1')
        plt.title('Team 1 vs. Winner Cross-tabulation')
        plt.tight_layout()
        plt.show()
        logging.info("Team1 vs winner cross-tabulation analysis completed.")
    except Exception as e:
        logging.error(f"Error analyzing team1 vs winner: {e}")
        raise

def check_null_values_heatmap(match_df):
    """Checks and visualizes null values in the dataframe using a heatmap."""
    try:
        logging.info("Checking for null values in match dataframe using heatmap.")
        plt.figure(figsize=(10, 6))
        sns.heatmap(match_df.isnull(), cmap='rainbow', yticklabels=False)
        plt.title("Null Values in Match Dataframe")
        plt.tight_layout()
        plt.show()
        logging.info("Null values check completed.")
    except Exception as e:
        logging.error(f"Error checking for null values: {e}")
        raise

if __name__ == '__main__':
    from src.data_collection import load_data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder where this script is located
    data_dir = os.path.join(BASE_DIR, 'data')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    matches_path = os.path.join(data_dir, 'matches.csv')

    # Load data using the paths constructed dynamically
    delivery_data, match_data = load_data(deliveries_path, matches_path)
    
    analyze_matches_per_season(match_data.copy())
    analyze_teams_per_season(match_data.copy())
    analyze_matches_by_team1(match_data.copy())
    analyze_toss_winner(match_data.copy())
    analyze_matches_result(match_data.copy())
    analyze_toss_match_winner(match_data.copy())
    # analyze_winner_of_toss_each_season(match_data.copy())
    analyze_top_player_of_the_match(match_data.copy())
    analyze_team1_vs_winner(match_data.copy())
    check_null_values_heatmap(match_data.copy())