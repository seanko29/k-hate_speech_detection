import tkinter as tk
import tkinter.font
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *

from datasets import load_dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, ElectraForPreTraining
from torch.utils.data import Dataset
import torch


from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
import numpy as np

# from khaters import CustomTrainer, KoElectraDataset
class CustomTrainer(Trainer):
    def train(self, *args, **kwargs):
        pass
    
    def evaluate(self, eval_dataset=None, *args, **kwargs):
        pass
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Retrieve logits from the model's output
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Fetch the labels
        labels = inputs.get('labels')

        # Calculate the cross-entropy loss
        # The logits are expected to be raw, unnormalized scores and labels should be of type long
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
# This is a placeholder for your model loading function
def load_model(path):
    # You should replace this with your actual model loading code
    print(f"Model loaded from {path}")

# # This is a placeholder for your model prediction function
# def classify_text():
#     prompt = text_entry.get("1.0", tk.END)
#     # You should replace this with your actual prediction code
#     # For example, label = model.predict(prompt)
#     label = "L1_hate"  # Placeholder
#     result_label.config(text=f"Predicted label: {id2label[label]}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)

# Define the label mapping
id2label = {0: 'L2_hate', 1: 'L1_hate', 2: 'offensive', 3: 'normal'}

# Placeholder for the model, tokenizer, and custom trainer
model = None
tokenizer = None
custom_trainer = None


def load_model():
    global model, tokenizer, custom_trainer
    model_path = filedialog.askdirectory(title="Select your trained model directory")
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained('beomi/korean-hatespeech-multilabel')
        model = ElectraForSequenceClassification.from_pretrained(model_path, num_labels=4)
        
        # Placeholder arguments, replace with actual arguments
        training_args = TrainingArguments(output_dir='./results', per_device_eval_batch_size=4)
        
        # Placeholder datasets, replace with actual datasets
        train_dataset = None
        test_dataset = None
        
        custom_trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        messagebox.showinfo("Model Loader", "Model loaded successfully from the directory!")


# Function to classify text
def classify_text():
    global custom_trainer, tokenizer
    sentence = text_entry.get("1.0", tk.END).strip()
    if not sentence:
        messagebox.showerror("Error", "Please enter some text to classify.")
        return
    
    if custom_trainer is None or tokenizer is None:
        messagebox.showerror("Error", "Please load the model first.")
        return
    
    # Tokenize the sentence and predict
    inputs = tokenizer(sentence, padding="max_length", truncation=True)
    outputs = custom_trainer.predict([inputs])
    predictions = np.argmax(outputs.predictions[0])
    id2label = {0: 'L2_hate', 1: 'L1_hate', 2: 'offensive', 3: 'normal'}
    print(predictions, 'predictions')
    # Display the result
    label = id2label[predictions]
    print(label, 'label')
    result_label.config(text=f'Sentence: {sentence}\nPredicted label: {label}')

# Create the main window
root = tk.Tk()
root.title("Hate Speech Classifier")
root.resizable(True, True)

# Create a text entry for the prompt
font = tkinter.font.Font(family='Malgun Gothic', size=22)


text_entry = Text(root, height=15, width=50, font=font)
text_entry.pack()

# Create a button to load the model
load_model_button = tk.Button(root, text="Load Model", command=load_model)
load_model_button.pack()

# Create a button to classify the text
classify_button = tk.Button(root, text="Classify Text", command=classify_text)
classify_button.pack()

# Create a label to display the result
result_label = tk.Label(root, text="Predicted label: None", font=font)
result_label.pack()

# Run the Tkinter event loop
root.mainloop()