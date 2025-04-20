import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def loadModel():

    # Load the fine-tuned model and tokenizer
    model_path = './fine_tuned_model'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory {model_path} not found. Please ensure the fine-tuned model is available.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Custom module for sentence embeddings (mean pooling)
    class CustomSentenceTransformer(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    sentence_model = CustomSentenceTransformer(model)
    # Load product data
    data_file = 'data/industrial_iot_skus.jsonl'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found. Please ensure the data file is available.")
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    # Preprocess to get input texts
    df['input_text'] = df['product_name'] + ' ' + df['manufacturer'] + ' ' + df['specifications'].apply(
        lambda specs: ' '.join([f"{key}: {value}" for key, value in specs.items()])
    )
    # Encode all product descriptions
    product_texts = df['input_text'].tolist()
    inputs = tokenizer(product_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        product_embeddings = sentence_model(inputs['input_ids'], inputs['attention_mask'])

    return tokenizer, sentence_model, df, product_embeddings