# Grammar Scoring Engine for Voice Samples

This project automatically scores and corrects grammar in transcribed voice samples using a fine-tuned T5 transformer model.

---

## 🚀 How to Run the Project

1. **Install the required packages**  
   Open your terminal or command prompt and run: 
   pip install transformers
   pip install torch
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install sentencepiece
   pip install transformers torch pandas

2. **Download the necessary model and dataset files**  
Since some files are too large for GitHub, download them from Google Drive:  
🔗 [Download All Large Files](https://drive.google.com/drive/folders/12RFuR7GXjnxBFrjSuw4nbOjSArp7ShFD?usp=drive_link)

After downloading, place them in the root directory of the project.

3. **Run the main application**  
Run this command to start: python app.py

---

## 🧠 What the Project Does

- Takes transcribed voice input  
- Uses a trained grammar correction model  
- Scores the grammar quality  
- Shows the corrected output  

---

## 📁 Main Project Files

- `app.py` – Main file to run the project
- `tokenizer.py` – Handles tokenizer setup
- `train_model.py` – Script to train the T5 model
- `inference.py` – Inference logic for model prediction
- `voice_to_text.py` – Converts voice to text
- `data_preprocessing.py` – Cleans and prepares data
- `inference_and_scoring.py` – Grammar scoring engine
- `preprocess_and_save_data.py` – Processes and saves prepped data
- `save_model.py` – Saves trained model

---

## 📦 Google Drive Contents

🔗 [Grammar Scoring Project Drive Folder](https://drive.google.com/drive/folders/12RFuR7GXjnxBFrjSuw4nbOjSArp7ShFD?usp=drive_link)

Includes:
- `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`
- `train_data.pkl`, `val_data.pkl`
- `trained_grammar_model/`
- Logs and results folders

---

## 👤 Author

**Macha Vardhan**  
📧 machavardhan05@gmail.com  
8897688191
🔗 [LinkedIn](https://linkedin.com/in/macha-vardhan177990298)

---

