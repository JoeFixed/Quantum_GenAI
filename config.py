from pydantic_settings import BaseSettings
import streamlit as st
class Settings(BaseSettings):
    TRANSLATION_MODEL_AR_EN: str = "Helsinki-NLP/opus-mt-ar-en"
    TRANSLATION_MODEL_EN_AR: str = "Helsinki-NLP/opus-mt-en-ar"
    SUMMARIZER_MODEL_PATH: str = "facebook/bart-large-cnn"
    NER_MODEL_PATH: str = "marefa-nlp/marefa-ner"
    IMAGE_EXTENSIONS: str = "png,jpg,jpeg"
    GPT_API_KEY: str = st.secrets["API_Key"]

    class Config:
        env_file = ".env"
