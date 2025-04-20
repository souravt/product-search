from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

from model import loadModel

# Initialize FastAPI app
app = FastAPI(title="Industrial IoT Product Search API")

# Define request model
class SearchRequest(BaseModel):
    query: str

# Define response model
class SearchResponse(BaseModel):
    products: list
    explanations: list

tokenizer, sentence_model, df, product_embeddings = loadModel()


# Function to extract key attributes from query
def extract_query_attributes(query):
    query = query.lower()
    attributes = {
        'category': None,
        'application': None,
        'location': None,
        'features': []
    }
    if 'water meter' in query:
        attributes['category'] = 'Water Meter'
    if 'residential' in query or 'apartment' in query:
        attributes['application'] = 'Residential'
    if 'bengaluru' in query:
        attributes['location'] = 'Bengaluru'
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
    filtered_df = filtered_df[filtered_df['status'].isin(['Available', 'Coming Soon'])]
    if attributes['features']:
        for feature in attributes['features']:
            filtered_df = filtered_df[filtered_df['features'].apply(lambda x: feature in x)]
    return filtered_df

# Function to search products
def search_products(query, top_k=3):
    attributes = extract_query_attributes(query)
    filtered_df = filter_products(df, attributes)
    if filtered_df.empty:
        return [], []

    filtered_indices = filtered_df.index.tolist()
    filtered_embeddings = product_embeddings[filtered_indices]

    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        query_embedding = sentence_model(query_input['input_ids'], query_input['attention_mask'])

    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_filtered_indices = filtered_df.index[top_indices]
    top_products = df.loc[top_filtered_indices]

    products_list = top_products[[
        'sku', 'product_name', 'manufacturer', 'category', 'application', 'features', 'specifications', 'status'
    ]].to_dict(orient='records')

    explanations = []
    for idx in top_filtered_indices:
        product = df.loc[idx]
        explanation = f"{product['product_name']} by {product['manufacturer']} is recommended because it is a {product['category']} designed for {product['application']} applications. "
        if attributes['location'] == 'Bengaluru':
            explanation += "It is suitable for Bengaluru's climate with features like "
        explanation += f"Features: {', '.join(product['features'])}. Specifications: {product['specifications']}. Status: {product['status']}."
        explanations.append(explanation)

    return products_list, explanations

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    products, explanations = search_products(request.query)
    if not products:
        return SearchResponse(products=[], explanations=["No matching products found."])
    return SearchResponse(products=products, explanations=explanations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)