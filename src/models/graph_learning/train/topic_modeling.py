from bertopic import BERTopic # type: ignore
from typing import Text
import os
from transformers import AutoModel, AutoTokenizer # type: ignore
import torch # type: ignore
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer

from src.models.vector_db.commons.input_loader import InputLoader

class TopicClass:
    
    def __init__(
        self,
        encoder_model: Text,
        inference_folder:  Text,
        output_pickle_file: Text,
        device: Text
        ) -> None:
        self.encoder_model = encoder_model
        self.inference_folder = inference_folder
        self.input_loader = InputLoader()
        inference_datapoints = self._load_all_input_from_dir(inference_folder)
        self.all_paragraphs = self._capture_all_paras(inference_datapoints=inference_datapoints)
        self.output_pickle_file = output_pickle_file
        if os.path.isdir(self.output_pickle_file):
            raise ValueError(f"The output path {self.output_pickle_file} is a directory. Please specify a valid file path ending with .pkl or other filename extension.")
        if os.path.basename(self.output_pickle_file) == "":
            raise ValueError(f"The output path {self.output_pickle_file} does not specify a file name. Please provide a valid file path.")
        if not os.path.exists(os.path.dirname(self.output_pickle_file)):
            os.makedirs(os.path.dirname(self.output_pickle_file))
        self.device = torch.device(device)

    def _load_all_input_from_dir(self, input_data_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        total_inference_datapoints = []
        for file in files:
            individual_datapoints = self.input_loader.load_data(data_file=file)
            total_inference_datapoints.extend(individual_datapoints) # type: ignore
        return total_inference_datapoints
    
    # def _capture_all_paras(self, inference_datapoints):
    #     total_paragraphs = []
    #     links = set()
    #     for item in inference_datapoints:
    #         if item["link"] in links:
    #             continue
    #         links.add(item["link"])
    #         total_paragraphs.extend(["\n".join(paras) for paras in item["all_paragraphs"]])
            
    #     return total_paragraphs
    def _capture_all_paras(self, inference_datapoints):
        total_paragraphs = []
        links = set()
        
        # Initialize CountVectorizer for stop word removal
        vectorizer_model = CountVectorizer(stop_words="english")
        
        for item in inference_datapoints:
            if item["link"] in links:
                continue
            links.add(item["link"])
            
            # Flatten paragraphs
            for paras in item["all_paragraphs"]:
                filtered_paras = []
                for para in paras:
                    # Use CountVectorizer to tokenize and remove stop words
                    tokens = vectorizer_model.build_analyzer()(para)
                    filtered_paras.append(" ".join(tokens))
                total_paragraphs.append("\n".join(filtered_paras))
        
        return total_paragraphs
    
    def _get_embeddings(self, docs, batch_size=1024):
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        model = AutoModel.from_pretrained(self.encoder_model).to(self.device)
        model.eval()
        
        all_embeddings = []
        
        with tqdm(total=len(docs), desc="Generating embeddings", unit="doc") as pbar:
            for start_idx in range(0, len(docs), batch_size):
                end_idx = min(start_idx + batch_size, len(docs))
                batch_docs = docs[start_idx:end_idx]
                
                encoded_input = tokenizer(
                    batch_docs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
                pbar.update(len(batch_docs))
        
        return np.vstack(all_embeddings)
        
    def get_topics(self, nr_topics=None):
        embeddings = self._get_embeddings(self.all_paragraphs)
        topic_model = BERTopic(embedding_model=self._get_embeddings)
        topics, probs = topic_model.fit_transform(self.all_paragraphs, embeddings)
        # print(topics, probs)
        if nr_topics is not None:
            topic_model.reduce_topics(self.all_paragraphs, nr_topics=nr_topics)
        topics_dict = topic_model.get_topic_info().to_dict(orient="list")
        topic_info = topic_model.get_topic_info()
        json_file = self.output_pickle_file.replace(".pkl", ".json")
        with open(json_file, "w") as f:
            json.dump(topics_dict, f, indent=4, ensure_ascii=False)
        
        with open(self.output_pickle_file, "wb") as f:
            pickle.dump(topic_model, f)
        # print(topic_info)

    def get_topic_embeddings(self, topic_model):
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        model = AutoModel.from_pretrained(self.encoder_model).to(self.device)
        model.eval()

        topic_embeddings = {}
        for topic_id, terms in topic_model.get_topics().items():
            if topic_id == -1:
                continue
            term_strings = [term[0] for term in terms]
            encoded_input = tokenizer(
                term_strings,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                model_output = model(**encoded_input)
                term_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()

            topic_embedding = np.mean(term_embeddings, axis=0)
            topic_embeddings[topic_id] = topic_embedding

        return topic_embeddings

if __name__ == "__main__":
    language = 'all'
    train_data_folder = f'input/train_infer/{language}/new_split/train_test_val'
    encoder_model = "bert-base-uncased"
    
    # output_topics_file = f"input/topics/{language}/topics.pkl"
    output_topics_file = f"/srv/upadro/embeddings/topics/{language}/topics.pkl"
    device = "cuda:3"
    topic_instance = TopicClass(encoder_model, train_data_folder, output_topics_file, device)
    topic_instance.get_topics(nr_topics=17)

    # Load the topic model and get topic embeddings
    with open(output_topics_file, "rb") as f:
        loaded_topic_model = pickle.load(f)

    topic_embeddings = topic_instance.get_topic_embeddings(loaded_topic_model)
    
    with open(f"/srv/upadro/embeddings/topics/{language}/topic_embeddings.pkl", "wb") as f:
        pickle.dump(topic_embeddings, f)
    print("Topic embeddings saved.")
