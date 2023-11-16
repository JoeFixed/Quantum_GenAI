from transformers import MarianMTModel, MarianTokenizer
import torch
# spacy download en_core_web_sm
# Load translation models
def load_translation_models():
    model_ar_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    tokenizer_ar_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    model_en_ar = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    tokenizer_en_ar = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    
    return model_ar_en, tokenizer_ar_en, model_en_ar, tokenizer_en_ar

# Perform translation
def perform_translation(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = ' '.join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translated_text
