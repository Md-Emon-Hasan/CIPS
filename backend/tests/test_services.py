from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from backend.app.services import preprocessing
from backend.app.services.data_collection import load_data
from backend.app.services.model_evaluation import (
    evaluate_classification_model,
    evaluate_model,
    train_and_evaluate,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

# --- Tests for data_collection.py ---


def test_load_data_success():
    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame({"a": [1, 2]})
        mock_read_csv.return_value = mock_df
        d, m = load_data("dummy_delivery.csv", "dummy_match.csv")
        assert d.equals(mock_df)
        assert m.equals(mock_df)


def test_load_data_file_not_found():
    with patch("pandas.read_csv", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_data("dummy_delivery.csv", "dummy_match.csv")


def test_load_data_generic_exception():
    with patch("pandas.read_csv", side_effect=Exception("Generic error")):
        with pytest.raises(Exception, match="Generic error"):
            load_data("dummy_delivery.csv", "dummy_match.csv")


# --- Tests for preprocessing.py ---


@pytest.fixture
def sample_delivery_df():
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2],
            "inning": [1, 2, 1],
            "batting_team": ["Team A", "Team B", "Team A"],
            "bowling_team": ["Team B", "Team A", "Team B"],
            "over": [1, 2, 1],
            "ball": [1, 2, 1],
            "batsman": ["P1", "P2", "P1"],
            "non_striker": ["P2", "P1", "P2"],
            "bowler": ["B1", "B2", "B1"],
            "is_super_over": [0, 0, 0],
            "wide_runs": [0, 0, 0],
            "bye_runs": [0, 0, 0],
            "legbye_runs": [0, 0, 0],
            "noball_runs": [0, 0, 0],
            "penalty_runs": [0, 0, 0],
            "batsman_runs": [1, 4, 6],
            "extra_runs": [0, 0, 0],
            "total_runs": [1, 4, 6],
            "player_dismissed": [np.nan, "P2", np.nan],
            "dismissal_kind": [np.nan, "bowled", np.nan],
            "fielder": [np.nan, np.nan, np.nan],
        }
    )


@pytest.fixture
def sample_match_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "season": [2020, 2021],
            "city": ["Mumbai", "Chennai"],
            "date": ["2020-01-01", "2021-01-01"],
            "team1": ["Team A", "Team C"],
            "team2": ["Team B", "Team D"],
            "toss_winner": ["Team A", "Team D"],
            "toss_decision": ["bat", "field"],
            "result": ["normal", "normal"],
            "dl_applied": [0, 1],
            "winner": ["Team A", "Team D"],
            "win_by_runs": [10, 0],
            "win_by_wickets": [0, 5],
            "player_of_match": ["P1", "P3"],
            "venue": ["Stadium A", "Stadium B"],
            "umpire1": ["U1", "U3"],
            "umpire2": ["U2", "U4"],
            "umpire3": [np.nan, np.nan],
        }
    )


def test_calculate_inning_runs(sample_delivery_df):
    df = preprocessing.calculate_inning_runs(sample_delivery_df)
    assert "total_runs" in df.columns
    assert len(df) == 3  # 1-1, 1-2, 2-1


def test_filter_first_innings():
    df = pd.DataFrame({"match_id": [1, 1], "inning": [1, 2], "total_runs": [100, 90]})
    res = preprocessing.filter_first_innings(df)
    assert len(res) == 1
    assert res["inning"].iloc[0] == 1


@patch("matplotlib.pyplot.show")
def test_visualize_total_runs_distribution(mock_show):
    df = pd.DataFrame({"total_runs": [100, 150, 200]})
    preprocessing.visualize_total_runs_distribution(df)
    mock_show.assert_called_once()


def test_merge_total_runs_with_match(sample_match_df):
    total_score = pd.DataFrame({"match_id": [1], "total_runs": [100]})
    res = preprocessing.merge_total_runs_with_match(sample_match_df, total_score)
    assert len(res) == 1
    assert "total_runs" in res.columns


def test_filter_teams():
    df = pd.DataFrame({"team1": ["Team A", "Team X"], "team2": ["Team B", "Team Y"]})
    teams = ["Team A", "Team B"]
    res = preprocessing.filter_teams(df, teams)
    assert len(res) == 1
    assert res.iloc[0]["team1"] == "Team A"


def test_standardize_team_names():
    df = pd.DataFrame(
        {
            "team1": ["Deccan Chargers", "Delhi Daredevils"],
            "team2": ["Delhi Daredevils", "Deccan Chargers"],
        }
    )
    res = preprocessing.standardize_team_names(df)
    assert (res["team1"] == "Sunrisers Hyderabad").any()
    assert (res["team1"] == "Delhi Capitals").any()


def test_filter_dl_applied():
    df = pd.DataFrame({"dl_applied": [0, 1, 0]})
    res = preprocessing.filter_dl_applied(df)
    assert len(res) == 2


def test_select_relevant_match_columns(sample_match_df):
    sample_match_df["total_runs"] = 100
    sample_match_df["match_id"] = sample_match_df["id"]  # Ensure match_id exists
    res = preprocessing.select_relevant_match_columns(sample_match_df)
    assert list(res.columns) == ["match_id", "city", "winner", "total_runs"]
    # Note: match_id is actually 'id' in sample_match_df, need to check function logic
    # function does: relevant_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
    # But input match_df usually has 'id' merged?
    # Wait, in main logic: match_runs_df = match_df.merge(total_score_df, left_on='id', right_on="match_id")
    # So match_runs_df has both 'id' and 'match_id'.
    # I should ensure input has match_id.

    df_input = pd.DataFrame(
        {
            "match_id": [1],
            "city": ["City"],
            "winner": ["Team A"],
            "total_runs": [100],
            "other": [1],
        }
    )
    res = preprocessing.select_relevant_match_columns(df_input)
    assert "other" not in res.columns


def test_merge_match_delivery(sample_delivery_df):
    match_df = pd.DataFrame({"match_id": [1], "city": ["Mumbai"]})
    res = preprocessing.merge_match_delivery(match_df, sample_delivery_df)
    # match_id 1 is in delivery df (2 rows)
    assert len(res) == 2
    assert "city" in res.columns


def test_filter_second_innings(sample_delivery_df):
    res = preprocessing.filter_second_innings(sample_delivery_df)
    assert len(res) == 1
    assert res["inning"].iloc[0] == 2


def test_calculate_cumulative_runs():
    df = pd.DataFrame({"match_id": [1, 1], "total_runs_y": [1, 4]})
    res = preprocessing.calculate_cumulative_runs(df)
    assert list(res["current_score"]) == [1, 5]


def test_calculate_runs_left():
    df = pd.DataFrame({"total_runs_x": [100, 100], "current_score": [10, 20]})
    res = preprocessing.calculate_runs_left(df)
    # 100 - 10 + 1 = 91? function: total_runs_x - current_score + 1
    assert list(res["runs_left"]) == [91, 81]


def test_calculate_balls_left():
    df = pd.DataFrame({"over": [0, 0], "ball": [1, 6]})
    res = preprocessing.calculate_balls_left(df)
    # 126 - (0*6 + 1) = 125
    # 126 - (0*6 + 6) = 120
    assert list(res["balls_left"]) == [125, 120]


def test_handle_player_dismissals():
    df = pd.DataFrame({"player_dismissed": [np.nan, "Player", "0", 0]})
    res = preprocessing.handle_player_dismissals(df)
    # nan -> 0 -> 0
    # 'Player' -> 1
    # '0' -> 1 (str)
    # 0 -> 0
    assert list(res["player_dismissed"]) == [0, 1, 1, 0]


def test_calculate_wickets_left():
    df = pd.DataFrame({"match_id": [1, 1], "player_dismissed": [1, 0]})
    res = preprocessing.calculate_wickets_left(df)
    # cumsum: 1, 1
    # 10 - 1 = 9
    # 10 - 1 = 9
    assert list(res["wickets"]) == [9, 9]


def test_calculate_current_run_rate():
    df = pd.DataFrame({"current_score": [60], "balls_left": [60]})
    res = preprocessing.calculate_current_run_rate(df)
    # 60 * 6 / (120 - 60) = 360 / 60 = 6.0
    assert res["crr"].iloc[0] == 6.0


def test_calculate_required_run_rate():
    df = pd.DataFrame({"runs_left": [60], "balls_left": [60]})
    res = preprocessing.calculate_required_run_rate(df)
    # 60 * 6 / 60 = 6.0
    assert res["rrr"].iloc[0] == 6.0


def test_reset_dataframe_index():
    df = pd.DataFrame({"a": [1]}, index=[5])
    res = preprocessing.reset_dataframe_index(df)
    assert res.index[0] == 0


def test_create_winner_column():
    df = pd.DataFrame(
        {"winner": ["Team A", "Team B"], "batting_team": ["Team A", "Team A"]}
    )
    res = preprocessing.create_winner_column(df)
    assert list(res["winner"]) == [1, 0]


def test_select_final_columns():
    df = pd.DataFrame(
        {
            "match_id": [1],
            "batting_team": ["A"],
            "bowling_team": ["B"],
            "city": ["C"],
            "runs_left": [1],
            "balls_left": [1],
            "wickets": [1],
            "total_runs_x": [1],
            "crr": [1],
            "rrr": [1],
            "winner": [1],
            "extra": [0],
        }
    )
    res = preprocessing.select_final_columns(df)
    assert "extra" not in res.columns


def test_shuffle_dataframe():
    df = pd.DataFrame({"a": range(10)})
    res = preprocessing.shuffle_dataframe(df)
    assert len(res) == 10
    # difficult to test randomness deterministically without seed, but function uses seed 42


def test_standardize_final_team_names():
    df = pd.DataFrame(
        {"batting_team": ["Deccan Chargers"], "bowling_team": ["Delhi Daredevils"]}
    )
    res = preprocessing.standardize_final_team_names(df)
    assert res["batting_team"].iloc[0] == "Sunrisers Hyderabad"


def test_handle_missing_city():
    df = pd.DataFrame(
        {
            "city": [0, "Mumbai"],
            "batting_team": ["Team A", "Team B"],
            "bowling_team": ["Team B", "Team A"],
        }
    )
    cities = {"Team A": "City A", "Team B": "City B"}
    # city 0 -> should use team home city
    # mocking random to control which team is picked?
    # function uses random.randint(0, 1)

    with patch("random.randint", return_value=0):
        # 0 -> batting_team -> Team A -> City A
        res = preprocessing.handle_missing_city(df, cities)
        assert res["city"].iloc[0] == "City A"


def test_drop_na_values():
    df = pd.DataFrame({"a": [1, np.nan]})
    res = preprocessing.drop_na_values(df)
    assert len(res) == 1


def test_remove_zero_balls_left():
    df = pd.DataFrame({"balls_left": [0, 10]})
    res = preprocessing.remove_zero_balls_left(df)
    assert len(res) == 1
    assert res["balls_left"].iloc[0] == 10


def test_drop_match_id():
    df = pd.DataFrame({"match_id": [1], "a": [2]})
    res = preprocessing.drop_match_id(df)
    assert "match_id" not in res.columns


def test_split_data():
    df = pd.DataFrame({"winner": [0, 1, 0, 1, 0], "feat1": [1, 2, 3, 4, 5]})
    # split_data drops winner from X
    X_train, X_test, y_train, y_test = preprocessing.split_data(df, test_size=0.2)
    assert len(X_train) + len(X_test) == 5
    assert "winner" not in X_train.columns


# --- Tests for model_evaluation.py ---


def test_evaluate_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([10, 20])
    X_test = [[1], [2]]
    y_test = [10, 20]
    # R2=1.0, MAE=0
    metrics = evaluate_model("Test", mock_model, X_test, y_test)
    assert metrics[0] == 1.0  # R2


def test_evaluate_classification_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1])
    X_test = [[1], [2]]
    y_test = [0, 1]
    # R2=1.0, MAE=0
    metrics = evaluate_classification_model("Test Cls", mock_model, X_test, y_test)
    assert metrics[-1] == 1.0  # Accuracy


@patch("backend.app.services.model_evaluation.pickle.dump")
def test_train_and_evaluate_regressor(mock_dump, tmp_path):
    # Need multiple categories to ensure OneHotEncoder doesn't drop everything
    X_train = pd.DataFrame(
        {
            "batting_team": ["A", "A"],
            "bowling_team": ["B", "B"],
            "city": ["C", "D"],  # Two cities
        }
    )
    y_train = [1, 2]
    X_test = X_train
    y_test = y_train
    model = LinearRegression()

    save_path = tmp_path / "dummy.pkl"
    pipe = train_and_evaluate(
        "Linear", model, X_train, X_test, y_train, y_test, save_path=str(save_path)
    )
    mock_dump.assert_called_once()
    assert pipe is not None


@patch("backend.app.services.model_evaluation.pickle.dump")
def test_train_and_evaluate_classifier(mock_dump, tmp_path):
    X_train = pd.DataFrame(
        {"batting_team": ["A", "A"], "bowling_team": ["B", "B"], "city": ["C", "D"]}
    )
    y_train = [0, 1]
    X_test = X_train
    y_test = y_train
    model = LogisticRegression()

    save_path = tmp_path / "dummy.pkl"
    train_and_evaluate(
        "Logistic", model, X_train, X_test, y_train, y_test, save_path=str(save_path)
    )
    mock_dump.assert_called_once()


def test_load_data_match_fail():
    with patch("pandas.read_csv") as mock_read_csv:
        # First call succeeds (delivery), second raises (match)
        mock_read_csv.side_effect = [pd.DataFrame({"a": [1]}), FileNotFoundError]
        with pytest.raises(FileNotFoundError):
            load_data("dummy_delivery.csv", "dummy_match.csv")


def test_load_data_match_generic_fail():
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = [pd.DataFrame({"a": [1]}), Exception("Generic")]
        with pytest.raises(Exception, match="Generic"):
            load_data("dummy_delivery.csv", "dummy_match.csv")


@patch("backend.app.services.model_evaluation.pickle.dump")
def test_save_model_exception(mock_dump, tmp_path):
    mock_dump.side_effect = Exception("Save failed")
    X_train = pd.DataFrame(
        {"batting_team": ["A", "A"], "bowling_team": ["B", "B"], "city": ["C", "D"]}
    )
    y_train = [1, 2]
    X_test = X_train
    y_test = y_train
    model = LinearRegression()

    # Should catch exception and log, not raise (based on code inspection lines 106-110 calls try...except logging)
    # The return pipe happens after.
    save_path = tmp_path / "dummy.pkl"
    pipe = train_and_evaluate(
        "Linear", model, X_train, X_test, y_train, y_test, save_path=str(save_path)
    )
    assert pipe is not None


def test_evaluate_model_exception():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        evaluate_model("ErrorModel", MagicMock(side_effect=ValueError), [], [])


def test_evaluate_classification_model_exception():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        evaluate_classification_model(
            "ErrorClsModel", MagicMock(side_effect=ValueError), [], []
        )


def test_train_and_evaluate_exception():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        # Pass None as model to trigger attribute error or similar, or mock
        train_and_evaluate("ErrorTrain", None, [], [], [], [])


# --- Tests for preprocessing.py exceptions ---


def test_calculate_inning_runs_exception():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_inning_runs(None)


def test_preprocessing_exceptions():
    # Helper to test simple exception wrapping functions
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.filter_first_innings(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.merge_total_runs_with_match(None, None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.filter_teams(None, None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.standardize_team_names(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.filter_dl_applied(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.select_relevant_match_columns(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.merge_match_delivery(None, None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.filter_second_innings(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_cumulative_runs(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_runs_left(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_balls_left(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.handle_player_dismissals(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_wickets_left(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_current_run_rate(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.calculate_required_run_rate(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.visualize_total_runs_distribution(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.reset_dataframe_index(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.drop_na_values(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.remove_zero_balls_left(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.drop_match_id(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.create_winner_column(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.select_final_columns(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.shuffle_dataframe(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.standardize_final_team_names(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.handle_missing_city(None, None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preprocessing.split_data(None)
