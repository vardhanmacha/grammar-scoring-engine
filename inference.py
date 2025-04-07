from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the fine-tuned model and tokenizer from local folder
model = T5ForConditionalGeneration.from_pretrained("grammar_scoring_model", local_files_only=True)
tokenizer = T5Tokenizer.from_pretrained("grammar_scoring_model", local_files_only=True)

# Function to correct grammar
def correct_grammar(text):
    input_text = f"grammar: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove "Grammatik:" or "grammar:" prefix from output (case-insensitive)
    if corrected.lower().startswith("grammatik:"):
        corrected = corrected[len("Grammatik:"):].strip()
    elif corrected.lower().startswith("grammar:"):
        corrected = corrected[len("grammar:"):].strip()

    return corrected

# Example usage
text = "he go to school every day"
corrected_text = correct_grammar(text)

print("Original:", text)
print("Corrected:", corrected_text)
