import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "CIPS - Cricket IPL Prediction System"
    API_V1_STR: str = "/api/v1"
    BASE_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    MODEL_PATH: str = os.path.join(BASE_DIR, "app", "ml_model", "pipe.pkl")
    DATABASE_URL: str = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'cips.db')}"

    model_config = {"case_sensitive": True}


settings = Settings()
os.makedirs(settings.LOGS_DIR, exist_ok=True)
