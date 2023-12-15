import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import load_metric

# Define the label mapping
id2label = {0: 'L2_hate', 1: 'L1_hate', 2: 'offensive', 3: 'normal'}

# Placeholder for the model and tokenizer
model = None
tokenizer = None

# Placeholder for compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)

# Function to load model and tokenizer
def load_model():
    global model, tokenizer
    model_path = filedialog.askopenfilename(title="Select your trained model")
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
        messagebox.showinfo("Model Loader", "Model and tokenizer loaded successfully!")

# Function to classify text
def classify_text():
    global model, tokenizer
    if tokenizer is None or model is None:
        messagebox.showerror("Error", "Please load a model first.")
        return
    
    sentence = text_entry.get("1.0", tk.END).strip()
    if not sentence:
        messagebox.showerror("Error", "Please enter some text to classify.")
        return
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=1)
    
    label = id2label[predictions[0]]
    result_label.config(text=f'Sentence: {sentence}\nPredicted label: {label}')

# Create the main window
root = tk.Tk()
root.title("Text Classifier")

# Create a text entry for the prompt
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

# Create a button to load the model
load_model_button = tk.Button(root, text="Load Model", command=load_model)
load_model_button.pack()

# Create a button to classify the text
classify_button = tk.Button(root, text="Classify Text", command=classify_text)
classify_button.pack()

# Create a label to display the result
result_label = tk.Label(root, text="Predicted label: None")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
