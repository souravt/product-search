from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd
import json

model_name = "sentence-transformers/all-MiniLM-L6-v2"  # or "mistralai/Mistral-7B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

peft_model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=100,
    no_cuda=True
)

# Load product data
with open('data/industrial_iot_skus.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Preprocess to create input texts
df['input_text'] = df['product_name'] + ' ' + df['manufacturer'] + ' ' + df['category']

# Create dataset
dataset = Dataset.from_pandas(df[['input_text']])


# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model and tokenizer
peft_model.save_pretrained('./peft_model')
tokenizer.save_pretrained('./peft_model')


