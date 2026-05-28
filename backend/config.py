from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    database_url: str = "postgresql://localhost/kisaan_db"
    hf_model_id: str = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openweather_api_key: str = ""
    langsmith_api_key: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""

    model_config = {"env_file": "backend/.env", "extra": "ignore"}


settings = Settings()
