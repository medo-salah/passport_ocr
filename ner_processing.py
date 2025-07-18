import spacy
import streamlit as st

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Model 'en_core_web_sm' not found. Run `python -m spacy download en_core_web_sm` to install.")
        st.stop()

def extract_with_ner(text, nlp):
    doc = nlp(text)
    fields = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            fields["Name (NER)"] = ent.text
        elif ent.label_ == "GPE":
            fields.setdefault("Address (NER)", ent.text)
        elif ent.label_ == "DATE":
            fields.setdefault("Date (NER)", ent.text)
        elif ent.label_ == "NORP":
            fields.setdefault("Nationality (NER)", ent.text)
    return fields