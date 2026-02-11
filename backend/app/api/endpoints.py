import logging
import pickle

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..core.config import settings
from ..db import get_session
from ..models import Prediction, PredictionCreate, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Load model
try:
    with open(settings.MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {settings.MODEL_PATH}: {e}")
    pipe = None


@router.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionCreate, session: Session = Depends(get_session)):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare data for prediction
        runs_left = input_data.target - input_data.score
        balls_left = 120 - (input_data.overs * 6)
        wickets_left = 10 - input_data.wickets
        crr = input_data.score / input_data.overs if input_data.overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        df = pd.DataFrame(
            {
                "batting_team": [input_data.batting_team],
                "bowling_team": [input_data.bowling_team],
                "city": [input_data.city],
                "runs_left": [runs_left],
                "balls_left": [balls_left],
                "wickets": [wickets_left],
                "total_runs_x": [input_data.target],
                "crr": [crr],
                "rrr": [rrr],
            }
        )

        # Predict
        result = pipe.predict_proba(df)
        win_prob_batting = round(result[0][1] * 100, 2)
        win_prob_bowling = round(result[0][0] * 100, 2)

        # Save to DB
        prediction_db = Prediction(
            **input_data.model_dump(),
            win_probability_batting=win_prob_batting,
            win_probability_bowling=win_prob_bowling,
        )
        session.add(prediction_db)
        session.commit()
        session.refresh(prediction_db)

        return PredictionResponse(
            **input_data.model_dump(),
            batting_team_probability=win_prob_batting,
            bowling_team_probability=win_prob_bowling,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
