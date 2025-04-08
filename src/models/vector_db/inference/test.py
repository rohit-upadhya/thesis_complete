import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
import json
from tqdm import tqdm
import os
from src.models.vector_db.commons.input_loader import InputLoader
def load_all_input_from_dir(input_data_path):
    """
    Load all JSON data files from a specified directory and return the combined data.
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    
    input_loader = InputLoader()  # Assuming InputLoader is a defined class to handle data loading
    total_inference_datapoints = []
    for file in files:
        individual_datapoints = input_loader.load_data(data_file=file)  # Load data from each JSON file
        total_inference_datapoints.extend(individual_datapoints)  # Combine data from all files
    
    return total_inference_datapoints

# data_dir = "src/models/single_datapoints/common"
data_dir = "input/inference_input/ukrainian/unique_query_test"

data = load_all_input_from_dir(data_dir)
print(f"Total data points loaded: {len(data)}")

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)  

def encode_texts(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def recall_at_k(actual, predicted, k):
    relevance = [1 if x in actual else 0 for x in predicted]
    r = np.asarray(relevance)[:k]
    return np.sum(r) / len(actual) if len(actual) > 0 else 0

recalls_2_percent = []
recalls_5_percent = []
recalls_10_percent = []

for i, datapoint in tqdm(enumerate(data), total=len(data)):
    case_name = datapoint['case_name']
    link = datapoint['link']
    
    actual_paragraph_numbers = datapoint.get('paragraph_numbers', [])
    
    queries = datapoint.get('query', [])
    
    if not queries:
        print(f"No queries found for case: {case_name}")
        continue

    combined_query = ", ".join(queries)

    paragraphs = ["\n".join(paras) for paras in datapoint['all_paragraphs']]
    paragraph_embeddings = encode_texts(paragraphs)

    embedding_dim = paragraph_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(paragraph_embeddings)
    print(f"Total embeddings added to FAISS index for {case_name}: {index.ntotal}")

    query_embedding = encode_texts([combined_query])[0] 
    
    query_embedding = query_embedding.reshape(1, -1).astype('float32')  
    distances, indices = index.search(query_embedding, index.ntotal)  
    
    
    top_2_percent = max(1, int(index.ntotal * 0.02)) 
    top_5_percent = max(1, int(index.ntotal * 0.05))  
    top_10_percent = max(1, int(index.ntotal * 0.1))
    
    top_2_percent_paragraphs = [indices[0][j] + 1 for j in range(top_2_percent)]
    top_5_percent_paragraphs = [indices[0][j] + 1 for j in range(top_5_percent)] 
    top_10_percent_paragraphs = [indices[0][j] + 1 for j in range(top_10_percent)]  
    
    recall_2 = recall_at_k(actual_paragraph_numbers, top_2_percent_paragraphs, k=len(actual_paragraph_numbers))
    recall_5 = recall_at_k(actual_paragraph_numbers, top_5_percent_paragraphs, k=len(actual_paragraph_numbers))
    recall_10 = recall_at_k(actual_paragraph_numbers, top_10_percent_paragraphs, k=len(actual_paragraph_numbers))
    
    recalls_2_percent.append(recall_2)
    recalls_5_percent.append(recall_5)
    recalls_10_percent.append(recall_10)

mean_recall_2_percent = np.mean(recalls_2_percent)
mean_recall_5_percent = np.mean(recalls_5_percent)
mean_recall_10_percent = np.mean(recalls_10_percent)

print(f"Mean Recall at 2%: {mean_recall_2_percent:.4f}")
print(f"Mean Recall at 5%: {mean_recall_5_percent:.4f}")
print(f"Mean Recall at 10%: {mean_recall_10_percent:.4f}")
