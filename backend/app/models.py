from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class PredictionBase(SQLModel):
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    wickets: int
    overs: float


class Prediction(PredictionBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    win_probability_batting: float
    win_probability_bowling: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionCreate(PredictionBase):
    pass


class PredictionResponse(PredictionBase):
    batting_team_probability: float
    bowling_team_probability: float
