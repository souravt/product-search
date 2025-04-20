from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Convert your JSONL to HuggingFace dataset format
dataset = load_dataset("json", data_files="/data/industrial_iot_skus.jsonl", split="train")

# Build prompts for training
def format_example(example):
    prompt = f"### Question: What is the best {example['category']} for {example['application']}?\n"
    prompt += f"### Answer: {example['product_name']} by {example['manufacturer']} with features like {', '.join(example['features'])}.\n"
    return {"text": prompt}

dataset = dataset.map(format_example)

tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

training_args = TrainingArguments(
    output_dir="./tinyllama-sku",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=100,
    logging_dir="./logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

quantized_model = AutoModelForCausalLM.from_pretrained(
    "./tinyllama-sku",
    quantization_config=bnb_config,
    device_map="auto"
)


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    input_ids = tokenizer(query.question, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=100)
    return {"answer": tokenizer.decode(output[0], skip_special_tokens=True)}

