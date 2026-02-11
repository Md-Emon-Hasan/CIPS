from unittest.mock import patch

import pandas as pd
import pytest
from backend.app.utils import eda


@pytest.fixture
def sample_match_df():
    return pd.DataFrame(
        {
            "Season": [2020, 2021, 2020],
            "team1": ["Team A", "Team B", "Team A"],
            "team2": ["Team C", "Team D", "Team C"],
            "toss_winner": ["Team A", "Team D", "Team A"],
            "winner": ["Team A", "Team D", "Team C"],
            "result": ["normal", "normal", "normal"],
            "player_of_match": ["P1", "P2", "P1"],
        }
    )


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.figure")
@patch("seaborn.countplot")
def test_analyze_matches_per_season(
    mock_countplot, mock_figure, mock_show, sample_match_df
):
    eda.analyze_matches_per_season(sample_match_df)
    mock_countplot.assert_called_once()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_teams_per_season(mock_show, sample_match_df):
    eda.analyze_teams_per_season(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_matches_by_team1(mock_show, sample_match_df):
    eda.analyze_matches_by_team1(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_toss_winner(mock_show, sample_match_df):
    eda.analyze_toss_winner(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_matches_result(mock_show, sample_match_df):
    eda.analyze_matches_result(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_toss_match_winner(mock_show, sample_match_df):
    eda.analyze_toss_match_winner(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_winner_of_toss_each_season(mock_show, sample_match_df):
    eda.analyze_winner_of_toss_each_season(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_top_player_of_the_match(mock_show, sample_match_df):
    eda.analyze_top_player_of_the_match(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_analyze_team1_vs_winner(mock_show, sample_match_df):
    eda.analyze_team1_vs_winner(sample_match_df)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_check_null_values_heatmap(mock_show, sample_match_df):
    eda.check_null_values_heatmap(sample_match_df)
    mock_show.assert_called_once()


def test_analyze_matches_per_season_exception():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_matches_per_season(None)  # Should raise error when accessing None


def test_eda_exceptions():
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_teams_per_season(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_matches_by_team1(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_toss_winner(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_matches_result(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_toss_match_winner(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_winner_of_toss_each_season(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_top_player_of_the_match(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.analyze_team1_vs_winner(None)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        eda.check_null_values_heatmap(None)
