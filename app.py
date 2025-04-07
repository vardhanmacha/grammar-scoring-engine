from voice_to_text import record_and_transcribe
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from nltk.metrics import edit_distance
from nltk.translate.bleu_score import sentence_bleu

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Grammar correction function
def correct_grammar(sentence):
    input_text = "fix grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Grammar score function using Levenshtein and BLEU
def grammar_score(original, corrected):
    # Levenshtein distance: Normalizing to be between 0 and 1
    lev_distance = edit_distance(original, corrected)
    max_len = max(len(original), len(corrected))
    lev_score = 1 - (lev_distance / max_len) if max_len > 0 else 1

    # BLEU score: Adjusted for more balanced output
    reference = [original.split()]  # BLEU expects tokenized sentences
    candidate = corrected.split()   # BLEU expects tokenized sentences
    bleu_score = sentence_bleu(reference, candidate)
    
    # Final score: A simple average of both scores
    score = (lev_score + bleu_score) / 2
    return score

# Streamlit UI
st.title("Grammar Scoring App")
st.write("Enter your sentence below to check and correct grammar:")

user_input = st.text_area("Your sentence")

if st.button("Check Grammar"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        corrected = correct_grammar(user_input)
        st.success("✅ Corrected Sentence:")
        st.write(corrected)

if st.button("Speak and Score"):
    spoken_text = record_and_transcribe()
    if spoken_text:
        st.write("You said:", spoken_text)
        corrected = correct_grammar(spoken_text)
        score = grammar_score(spoken_text, corrected)
        st.write("Corrected Sentence:", corrected)
        st.write("Grammar Score:", score)
