from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load Teacher and Student
teacher_model = AutoModelForCausalLM.from_pretrained("mistral-7b-instruct", device_map="auto")
student_model = AutoModelForCausalLM.from_pretrained("TinyLLaMA", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("TinyLLaMA")

# Load dataset: could be your product catalog Q&A with teacher responses
dataset = load_dataset("json", data_files="teacher_labeled_data.jsonl", split="train")

# Tokenize inputs & teacher outputs as targets
def preprocess(example):
    inputs = tokenizer(example["instruction"], return_tensors="pt", truncation=True, padding=True)
    labels = tokenizer(example["teacher_output"], return_tensors="pt", truncation=True, padding=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True)

# Define trainer (you can customize loss for distillation)
training_args = TrainingArguments(
    output_dir="./distilled-model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    learning_rate=2e-5,
    fp16=True
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
