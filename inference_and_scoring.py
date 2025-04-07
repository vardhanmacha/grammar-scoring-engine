import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

# Load dataset
df = pd.read_csv(r"C:\grammer project\archive\Grammar correction.csv")

# Check required columns
if not {'Ungrammatical Statement', 'Standard English'}.issubset(df.columns):
    raise ValueError("Dataset must contain 'Ungrammatical Statement' and 'Standard English' columns.")

# Load model and tokenizer
model_name = "vennify/t5-base-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Grammar correction function
def correct_grammar(sentence):
    input_text = "fix grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Inference and scoring
results = []
correct_count = 0
total = 0

LIMIT = 100  # ⬅️ Change this to however many rows you want to test

for i, row in df.iterrows():
    if i == LIMIT:
        break

    original = row["Ungrammatical Statement"]
    expected = row["Standard English"]
    corrected = correct_grammar(original)

    score = 1 if corrected.strip().lower() == expected.strip().lower() else 0
    correct_count += score
    total += 1

    print(f"\nOriginal: {original}")
    print(f"Expected: {expected}")
    print(f"Corrected: {corrected}")
    print(f"Score: {score}")

    results.append({
        "Original": original,
        "Expected": expected,
        "Corrected": corrected,
        "Score": score
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("results/inference_results.csv", index=False)

# Show final accuracy
accuracy = (correct_count / total) * 100
print(f"\nInference complete for {total} samples")
print(f"Grammar Correction Accuracy: {accuracy:.2f}%")


