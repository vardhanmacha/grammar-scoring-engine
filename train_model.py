import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load preprocessed data
with open("train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("val_data.pkl", "rb") as f:
    val_data = pickle.load(f)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize the data
def tokenize_data(example):
    input_enc = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=128)
    target_enc = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_dataset = Dataset.from_list(train_data).map(tokenize_data, batched=False)
val_dataset = Dataset.from_list(val_data).map(tokenize_data, batched=False)

# Define training arguments (✅ using eval_strategy instead of evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("trained_grammar_model")
tokenizer.save_pretrained("trained_grammar_model")



# Save the model and tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

model.save_pretrained("grammar_scoring_model")
tokenizer.save_pretrained("grammar_scoring_model")

print(" Model and tokenizer saved in 'grammar_scoring_model' folder.")
