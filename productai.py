import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import os

# Define the path to the local JSONL file
local_file = 'data/industrial_iot_skus.jsonl'

# Check if the file exists
if not os.path.exists(local_file):
    raise FileNotFoundError(f"The file {local_file} does not exist. Please ensure the data file is in the correct location.")

# Load the JSONL data from the local file
data = []
with open(local_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Preprocess the data
def preprocess_specifications(specs):
    return ' '.join([f"{key}: {value}" for key, value in specs.items()])

df['input_text'] = df.apply(
    lambda row: f"{row['product_name']} {row['manufacturer']} {preprocess_specifications(row['specifications'])}",
    axis=1
)
df['label'] = df['category']

# Encode the category labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df[['input_text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['input_text', 'label']])

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_encoder.classes_)
)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the adaptation
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # Target query and value layers in DistilBERT
    lora_dropout=0.1,
    bias="none",
)
peft_model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Small batch size for laptop
    per_device_eval_batch_size=8,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    no_cuda=True
)

# Define metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Initialize Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)


# Save the fine-tuned model
peft_model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Inference function
def predict_category(input_text):
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = peft_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0]

# Example prediction
sample_input = "sensorX® 50DN ABB pipe_diameter: 50 mm ip_rating: IP68 measurement_principle: Ultrasonic power_source: Solar temperature_range: -17°C to 54°C"
predicted_category = predict_category(sample_input)
print(f"Predicted Category: {predicted_category}")