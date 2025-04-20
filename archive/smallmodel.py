# Starter notebook to fine-tune a small LLM with LoRA on product data and deploy it

# Step 1: Install dependencies (Run this in your Colab or environment shell)
# !pip install transformers datasets peft accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import torch
import json

# Step 2: Load Base Model & Tokenizer (Choose a small model)
model_name = "microsoft/phi-2"  # Lightweight & performs well
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Add LoRA Adapter
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Step 4: Prepare Data (Assume product dataset is instruction-tuned format)
# Load dataset (replace with actual JSONL path or dataset)
data = []
with open("data/industrial_iot_skus_instruction.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        data.append({"text": f"### Instruction:\n{entry['instruction']}\n\n### Response:\n{entry['output']}"})

dataset = Dataset.from_list(data)

# Tokenization
MAX_LEN = 512
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir="lora_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True
)

# Step 6: Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# Step 7: Quantize the LoRA Model (4-bit for deployment)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
quantized_model = AutoModelForCausalLM.from_pretrained("lora_finetuned_model", quantization_config=bnb_config, device_map="auto")

# Step 8: Test Inference
prompt = "Which water meter is best for smart city apartment use in India?"
inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to("cuda")
out = quantized_model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# Step 9: Deploy via FastAPI (in a separate main.py file)
# from fastapi import FastAPI, Request
# import uvicorn

# app = FastAPI()

# @app.post("/ask")
# async def ask(request: Request):
#     data = await request.json()
#     prompt = data.get("query")
#     inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to("cuda")
#     outputs = quantized_model.generate(**inputs, max_new_tokens=150)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Optional: Push to Hugging Face Hub if you have an account
# model.push_to_hub("your-username/lora-product-model")
# tokenizer.push_to_hub("your-username/lora-product-model")
