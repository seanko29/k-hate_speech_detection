from datasets import load_dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, ElectraForPreTraining
from torch.utils.data import Dataset
import torch


from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
import numpy as np


# data call

data = load_dataset('humane-lab/K-HATERS')
train = data['train']
valid = data['validation']
test = data['test']

# initialize pre-trained model and tokenizer

id2label = {0: 'L2_hate', 1: 'L1_hate', 2: 'offensive', 3: 'normal'}
label2id = {'L2_hate': 0, 'L1_hate': 1, 'offensive': 2, 'normal': 3}
tokenizer = AutoTokenizer.from_pretrained('beomi/korean-hatespeech-multilabel')
model = ElectraForSequenceClassification.from_pretrained('beomi/korean-hatespeech-multilabel',
                                                         num_labels=4, 
                                                         id2label=id2label, 
                                                         label2id=label2id,
                                                         problem_type='single_label_classification')


train_encodings = tokenizer(train['text'], truncation=True, padding="max_length")
valid_encodings = tokenizer(valid['text'], truncation=True, padding="max_length")
test_encodings = tokenizer(test['text'], truncation=True, padding="max_length")

train_labels = train['label']
valid_labels = valid['label']
test_labels = test['label']



class CustomTrainer(Trainer):
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


class KoElectraDataset(Dataset):
    def __init__(self, encodings, labels, label2id):
        self.encodings = encodings
        self.labels = labels
        self.label2id = label2id

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label_str = self.labels[idx]
        label_id = self.label2id[label_str]  # Convert the string label to an integer ID
        item['labels'] = torch.tensor(label_id, dtype=torch.long)  # Ensure the label is a tensor of type long
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    

train_dataset = KoElectraDataset(train_encodings, train_labels, label2id)
valid_dataset = KoElectraDataset(valid_encodings, valid_labels, label2id)
test_dataset = KoElectraDataset(test_encodings, test_labels, label2id)


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
)


custom_trainer = CustomTrainer(
    model=model,  # make sure your model is for multi-class classification
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    # Add other parameters if needed
)

custom_trainer.train()
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')


predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
