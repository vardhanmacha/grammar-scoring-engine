from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Sample dataset
data = {
    "input_text": [
        "He go to school every day.",
        "She are playing in the garden.",
        "They is eating dinner."
    ],
    "target_text": [
        "He goes to school every day.",
        "She is playing in the garden.",
        "They are eating dinner."
    ]
}

# Create HuggingFace Dataset
dataset = Dataset.from_dict(data)

# Load pretrained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess function
def preprocess(example):
    input_text = "grammar: " + example["input_text"]
    target_text = example["target_text"]

    model_inputs = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            max_length=128,
            padding="max_length",
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, remove_columns=["input_text", "target_text"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_steps=10,
    remove_unused_columns=False,
    report_to="none"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("grammar_scoring_model")
tokenizer.save_pretrained("grammar_scoring_model")

print("Model and tokenizer saved in 'grammar_scoring_model' folder.")

