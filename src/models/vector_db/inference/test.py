import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
import json
from tqdm import tqdm
import os  # Required for directory traversal
from src.models.vector_db.commons.input_loader import InputLoader
# Load the JSON files from a given directory
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

# Example data directory paths
# data_dir = "src/models/single_datapoints/common"
data_dir = "input/inference_input/ukrainian/unique_query_test"

# Load data from the specified directory
data = load_all_input_from_dir(data_dir)
print(f"Total data points loaded: {len(data)}")

# Check if GPU is available and set device accordingly
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained BERT model and tokenizer, and move the model to GPU if available
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)  # Move model to GPU 'bert-base-multilingual-cased'

# Function to encode a list of paragraphs or queries using BERT
def encode_texts(texts):
    # Tokenize and encode, move inputs to the correct device
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU if available
    
    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Take the mean of the last hidden state for each input text and move to CPU
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Ensure the output is on CPU
    return embeddings

# Function to calculate recall at k
def recall_at_k(actual, predicted, k):
    relevance = [1 if x in actual else 0 for x in predicted]
    r = np.asarray(relevance)[:k]
    return np.sum(r) / len(actual) if len(actual) > 0 else 0

# Store individual recall results for each percentage
recalls_2_percent = []
recalls_5_percent = []
recalls_10_percent = []

# Iterate over each data point to create individual FAISS indices and calculate recall
for i, datapoint in tqdm(enumerate(data), total=len(data)):
    case_name = datapoint['case_name']
    link = datapoint['link']
    
    # Get the paragraph numbers from the dataset (1-indexed)
    actual_paragraph_numbers = datapoint.get('paragraph_numbers', [])
    
    # Use the provided queries in the data point and concatenate them into a single query
    queries = datapoint.get('query', [])
    
    # If there are no queries, skip this data point
    if not queries:
        print(f"No queries found for case: {case_name}")
        continue

    # Combine all queries into a single query string separated by commas
    combined_query = ", ".join(queries)

    # Encode all paragraphs for the current data point
    paragraphs = ["\n".join(paras) for paras in datapoint['all_paragraphs']]
    paragraph_embeddings = encode_texts(paragraphs)

    # Create a separate FAISS index for this data point
    embedding_dim = paragraph_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(paragraph_embeddings)
    print(f"Total embeddings added to FAISS index for {case_name}: {index.ntotal}")

    # Encode the combined query as a single query
    query_embedding = encode_texts([combined_query])[0]  # Single embedding for the combined query
    
    # Search the FAISS index for this query
    query_embedding = query_embedding.reshape(1, -1).astype('float32')  # Reshape for FAISS
    distances, indices = index.search(query_embedding, index.ntotal)  # Get all paragraphs ranked
    
    # Get the top percentages for each filtered result
    top_2_percent = max(1, int(index.ntotal * 0.02))  # Top 2% of results
    top_5_percent = max(1, int(index.ntotal * 0.05))  # Top 5% of results
    top_10_percent = max(1, int(index.ntotal * 0.1))  # Top 10% of results
    
    # Get the paragraph numbers for top percentages
    top_2_percent_paragraphs = [indices[0][j] + 1 for j in range(top_2_percent)]  # Convert to 1-indexed
    top_5_percent_paragraphs = [indices[0][j] + 1 for j in range(top_5_percent)]  # Convert to 1-indexed
    top_10_percent_paragraphs = [indices[0][j] + 1 for j in range(top_10_percent)]  # Convert to 1-indexed
    
    # Calculate individual recalls for each percentage
    recall_2 = recall_at_k(actual_paragraph_numbers, top_2_percent_paragraphs, k=len(actual_paragraph_numbers))
    recall_5 = recall_at_k(actual_paragraph_numbers, top_5_percent_paragraphs, k=len(actual_paragraph_numbers))
    recall_10 = recall_at_k(actual_paragraph_numbers, top_10_percent_paragraphs, k=len(actual_paragraph_numbers))
    
    # Store the individual recall values
    recalls_2_percent.append(recall_2)
    recalls_5_percent.append(recall_5)
    recalls_10_percent.append(recall_10)

# Calculate the mean recall for each percentage
mean_recall_2_percent = np.mean(recalls_2_percent)
mean_recall_5_percent = np.mean(recalls_5_percent)
mean_recall_10_percent = np.mean(recalls_10_percent)

# Print the overall recall scores
print(f"Mean Recall at 2%: {mean_recall_2_percent:.4f}")
print(f"Mean Recall at 5%: {mean_recall_5_percent:.4f}")
print(f"Mean Recall at 10%: {mean_recall_10_percent:.4f}")
