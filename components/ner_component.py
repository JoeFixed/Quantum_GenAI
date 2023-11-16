import streamlit as st
from spacy_streamlit import visualize_ner
import spacy

nlp = spacy.load('en_core_web_sm')

def show_ner_component(text):
    doc = nlp(text)
    st.subheader("Named Entity Recognition")
    visualize_ner(doc)
