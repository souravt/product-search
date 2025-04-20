from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

# Load the fine-tuned model and tokenizer
model_path = './fine_tuned_model'
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
with open('data/industrial_iot_skus.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Preprocess to get input texts
df['input_text'] = df['product_name'] + ' ' + df['manufacturer'] + ' ' + df['category']

# Encode all product descriptions
product_texts = df['input_text'].tolist()
inputs = tokenizer(product_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
with torch.no_grad():
    product_embeddings = sentence_model(inputs['input_ids'], inputs['attention_mask'])

# Function to search products
def search_products(query, top_k=10):
    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        query_embedding = sentence_model(query_input['input_ids'], query_input['attention_mask'])
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_products = df.iloc[top_indices]
    return top_products[['sku', 'product_name', 'manufacturer', 'category']]

# Interactive search loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    results = search_products(query)
    print(results)