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
df['input_text'] = df['product_name'] + ' ' + df['manufacturer'] + ' ' + df['specifications'].apply(
    lambda specs: ' '.join([f"{key}: {value}" for key, value in specs.items()])
)

# Encode all product descriptions
product_texts = df['input_text'].tolist()
inputs = tokenizer(product_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
with torch.no_grad():
    product_embeddings = sentence_model(inputs['input_ids'], inputs['attention_mask'])


# Function to extract key attributes from query
def extract_query_attributes(query):
    query = query.lower()
    attributes = {
        'category': None,
        'application': None,
        'location': None,
        'features': []
    }
    # Extract category
    if 'water meter' in query:
        attributes['category'] = 'Water Meter'
    # Extract application
    if 'residential' in query or 'apartment' in query:
        attributes['application'] = 'Residential'
    # Extract location
    if 'bengaluru' in query:
        attributes['location'] = 'Bengaluru'
    # Extract desired features
    if 'leak' in query:
        attributes['features'].append('Leak detection')
    if 'durable' in query or 'monsoon' in query:
        attributes['features'].append('IP68 rating')
    return attributes


# Function to filter products based on query attributes
def filter_products(df, attributes):
    filtered_df = df.copy()
    if attributes['category']:
        filtered_df = filtered_df[filtered_df['category'] == attributes['category']]
    if attributes['application']:
        filtered_df = filtered_df[filtered_df['application'] == attributes['application']]
    # Prioritize available products
    filtered_df = filtered_df[filtered_df['status'].isin(['Available', 'Coming Soon'])]
    # Filter by features if specified
    if attributes['features']:
        for feature in attributes['features']:
            filtered_df = filtered_df[filtered_df['features'].apply(lambda x: feature in x)]
    return filtered_df


# Function to search products
def search_products(query, top_k=3):
    # Extract query attributes
    attributes = extract_query_attributes(query)

    # Filter products based on attributes
    filtered_df = filter_products(df, attributes)
    if filtered_df.empty:
        return pd.DataFrame(), []

    # Encode query
    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        query_embedding = sentence_model(query_input['input_ids'], query_input['attention_mask'])

    # Get embeddings for filtered products
    filtered_indices = filtered_df.index.tolist()
    filtered_embeddings = product_embeddings[filtered_indices]

    # Compute similarities
    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

    # Get top-k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_filtered_indices = filtered_df.index[top_indices]
    top_products = df.loc[top_filtered_indices]

    # Generate explanations
    explanations = []
    for idx in top_filtered_indices:
        product = df.loc[idx]
        explanation = f"{product['product_name']} by {product['manufacturer']} is recommended because it is a {product['category']} designed for {product['application']} applications. "
        if attributes['location'] == 'Bengaluru':
            explanation += "It is suitable for Bengaluru's climate with features like "
        explanation += f"Features: {', '.join(product['features'])}. Specifications: {product['specifications']}. Status: {product['status']}."
        explanations.append(explanation)

    return top_products[['sku', 'product_name', 'manufacturer', 'category', 'application', 'features', 'specifications',
                         'status']], explanations


# Interactive search loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    results, explanations = search_products(query)
    if results.empty:
        print("No matching products found.")
    else:
        print("\nRecommended Products:")
        print(results.to_string(index=False))
        print("\nExplanations:")
        for i, explanation in enumerate(explanations, 1):
            print(f"{i}. {explanation}\n")