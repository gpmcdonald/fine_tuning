from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "SyMoNeuRaL API"
    debug: bool = True

settings = Settings()