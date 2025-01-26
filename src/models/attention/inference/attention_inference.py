from typing import Text, List, Dict, Any, Optional
import numpy as np # type: ignore
import os
import json
import math
from tqdm import tqdm # type: ignore
import warnings
warnings.filterwarnings("ignore")
import torch # type: ignore

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.encoder import Encoder
from src.models.vector_db.inference.val_encoder import ValEncoder
from src.models.vector_db.inference.faiss_vector_db import FaissVectorDB
from src.models.attention.encoders.attention_block import AttentionBlock

class Inference:
    def __init__(
            self, 
            inference_folder: Optional[Text] = 'input/train_infer/english/new_split/test', 
            inference_datapoints: Optional[List[Dict[str, Any]]] = None, 
            bulk_inference: bool = False,
            use_translations: bool = False,
            device: str = "cpu",
            language: str = "english",
            question_model_name_or_path: str = 'bert-base-multilingual-cased',
            ctx_model_name_or_path: str = 'bert-base-multilingual-cased',
            save_recall: bool = True,
            run_val: bool = False,
            tokenizer: Optional[str] = None,
            model: Optional[str] = None,
            device_: Optional[str] = None,
            attention_model: str = '/srv/upadro/models/graph/2024-12-13___all_gat___training/_final_model/graph_model.pt',
            comment: str = 'topic',
        ):
        self.use_all = False
        self.input_loader = InputLoader()
        self.file_base_name = language
        self.use_translations = use_translations
        self.run_val = run_val
        self.recall_file_name = f"recall_{attention_model.replace('/', '_')}_{'translated' if use_translations else 'not_translated'}_{inference_folder.split('/')[-1]}.json" # type: ignore
        self.language = language
        self.save_recall = save_recall
        if bulk_inference:
            inference_datapoints = self._load_all_input_from_dir(inference_folder)
            self.file_base_name = inference_folder.split("/")[-3] # type: ignore
        self._format_input(inference_datapoints)
        self.folder_name = inference_folder.split('/')[-1] # type: ignore
        if run_val:
            self.encoder = ValEncoder(question_model=model, ctx_model=model, question_tokenizer=tokenizer, ctx_tokenizer=tokenizer, device=device_)
        else:
            self.encoder = Encoder(device=device, question_model_name_or_path=question_model_name_or_path, ctx_model_name_or_path=ctx_model_name_or_path, use_dpr=False, use_roberta=False)
        self.faiss = FaissVectorDB()
        self.comment = comment
        self.attention_model = attention_model
        self.all_paragraph_encodings = []
        self.all_unique_keys = []
        self.query_paragraph_mapping = {}
        self.model_trained_language = question_model_name_or_path.split("_")[-4] if "training" in question_model_name_or_path else "base"
    
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
        print("self.run_val", self.run_val)
        if self.run_val:
            return total_inference_datapoints
        return total_inference_datapoints


    def _format_input(self, inference_datapoints):
        self.data_points = []
        
        translatons_query = {}
        if self.language != 'english' and self.use_translations:
            translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{self.language}.json")
        for idx, data in enumerate(inference_datapoints):
            key = data["link"]
            final_query = []
            if self.use_translations and self.language != 'english':
                for query_item in data["query"]:
                    for trdata_point in translatons_query: # type: ignore
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
            else:
                final_query = data["query"]
            
            unique_key = f"datapoint_{key}"
            self.data_points.append(
                {
                    "query": ", ".join(final_query),
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i+1}" for i in range(len(data["all_paragraphs"]))],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "link": key,
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
        print(self.data_points[0]["query"])
        print(inference_datapoints[0]["query"])
    
    def _get_attention_encodings(self, attention_model, encodings):
        
        encodings = encodings.unsqueeze(0)

        with torch.no_grad():
            updated, attn_weights = attention_model(encodings) 

        updated = updated.squeeze(0)
        updated_encodings = updated.cpu().numpy()
        return updated_encodings
    
    def _encode_all_paragraphs(self, attention_model, batch_size=512,):
        index_counter = 0
        paragraph_set = set()
        total_paragraphs = sum(len(points["all_paragraphs"]) for points in self.data_points if points["link"] not in paragraph_set)
        with tqdm(total=total_paragraphs, desc="Encoding paragraphs", unit="paragraph") as pbar:
            for idx, points in enumerate(self.data_points):
                if points["link"] not in paragraph_set:
                    all_paragraphs = points["all_paragraphs"]
                    num_paragraphs = len(all_paragraphs)
                    encoded_paragraphs = []

                    for start_idx in range(0, num_paragraphs, batch_size):
                        end_idx = min(start_idx + batch_size, num_paragraphs)
                        batch_paragraphs = all_paragraphs[start_idx:end_idx]
                        
                        encoded_batch = self.encoder.encode_ctx(batch_paragraphs).cpu().numpy()
                        encoded_paragraphs.append(encoded_batch)
                        pbar.update(end_idx - start_idx)

                    encoded_paragraphs = np.vstack(encoded_paragraphs)
                    encoded_paragraphs = torch.from_numpy(encoded_paragraphs)
                    encoded_paragraphs = self._get_attention_encodings(attention_model=attention_model, encodings=encoded_paragraphs)
                    
                    self.all_paragraph_encodings.append(encoded_paragraphs)
                    self.all_unique_keys.extend(points["unique_keys"])
                    self.query_paragraph_mapping[idx] = list(range(index_counter, index_counter + len(all_paragraphs)))
                    index_counter += len(all_paragraphs)
                    paragraph_set.add(points["link"])
                    
        all_encodings = np.vstack(self.all_paragraph_encodings)
        self.faiss.build_index(all_encodings, self.all_unique_keys)

    def _encode_query(self, query: Text):
        return self.encoder.encode_question([query]).cpu().numpy()
    
    
    def recall_at_k(self, actual, predicted, k):
        relevance = [1 if x in actual else 0 for x in predicted]
        r = np.asarray(relevance)[:k]
        return np.sum(r) / len(actual) if len(actual) > 0 else 0

    def calculate_recall(self, results):
        percentages = [2, 5, 10]
        recall = {}
        
        for percentage in percentages:
            r_at_percentage = []
            for idx, result in enumerate(results):
                query_data = self.data_points[idx]
                actual_relevant = set(query_data.get('paragraph_numbers', []))
                
                predicted_relevant = result[f'relevant_paragraphs']
                k = result[f"no_{percentage}"]
                recall_score = self.recall_at_k(actual_relevant, predicted_relevant, k)
                r_at_percentage.append(recall_score)
            
            recall[f"mean_recall_at_{percentage}_percentage"] = np.mean(np.asarray(r_at_percentage))
        
        return recall

    def _load_model(self, model_path, hidden_dim):
        """
        Load the trained graph model from a saved state dict.
        """
        model = AttentionBlock(embedding_dim=hidden_dim, no_heads=8)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
        
    def main(self):
        attention_model = self._load_model(model_path=self.attention_model, hidden_dim=768)
        self._encode_all_paragraphs(attention_model=attention_model, batch_size=512)
        results = []
        print(self.attention_model)
        for idx, points in enumerate(self.data_points):
            query_encodings = self._encode_query(points['query'])
            number_of_relevant_paragraph_2 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.02)), 1)
            number_of_relevant_paragraph_5 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.05)), 1)
            number_of_relevant_paragraph_10 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.1)), 1)            
            relevant_paragraph_keys = self.faiss.perform_search(
                query=query_encodings,
                datapoint = points,
            )
            results.append({
                "query": points["query"],
                "link": points["link"],
                "relevant_paragraphs": [int(paragraph.split("_")[-1]) for paragraph in relevant_paragraph_keys],
                "no_2": number_of_relevant_paragraph_2,
                "no_5": number_of_relevant_paragraph_5,
                "no_10": number_of_relevant_paragraph_10
            })
            
        recall = self.calculate_recall(results)
        print(f"Recall: {recall}")
        
        recall_data = {}
        if self.save_recall:
            recall_path = os.path.join('output/inference_outputs/new_splits/new_experiments/attention/',self.comment, self.model_trained_language, self.folder_name, self.recall_file_name)
            recall_dir = os.path.dirname(recall_path)
            os.makedirs(recall_dir, exist_ok=True)
            if os.path.exists(recall_path):
                with open(recall_path, 'r') as json_file:
                    recall_data = json.load(json_file)

            recall_data[self.file_base_name] = recall
            recall_data["model"] = self.attention_model
            with open(recall_path, 'w+') as json_file:
                json.dump(recall_data, json_file, indent=4, ensure_ascii=False)
        return recall

if __name__ == "__main__":
    # languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    # languages = ["russian"]
    # file = "val"
    
    # translations  = [True, False]
    # file = "val"
    # models = [
    #     ['bert-base-multilingual-cased', 'bert-base-multilingual-cased'],
    #     ['castorini/mdpr-tied-pft-msmarco', 'castorini/mdpr-tied-pft-msmarco'],
    #     ['castorini/mdpr-tied-pft-msmarco-ft-all', 'castorini/mdpr-tied-pft-msmarco-ft-all'],
    #     ['bert-base-uncased', 'bert-base-uncased']
    # ]

    
    # models = [
    #     ["facebook/dpr-question_encoder-single-nq-base",
    #     # "facebook/dpr-question_encoder-single-nq-base"]
         
    #      "facebook/dpr-ctx_encoder-single-nq-base"]
    # ]
    
    files = [
            'unique_query', 
            'test'
        ]
    # files = ['test']
    models = [
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_0/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_5/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_10/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_15/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_20/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_25/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_30/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_35/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_2/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    [
        "castorini/mdpr-tied-pft-msmarco",
        "castorini/mdpr-tied-pft-msmarco",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_base_training/checkpoints/epoch_39/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_base"
    ],
    
    
    
    
    
    
    
    
    
    
    
    
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_0/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_5/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_10/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_15/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_20/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_25/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_30/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_35/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_2/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    [
       "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
        "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
        "/srv/upadro/models/new_expt/attention/2025-01-11___all_new_attention_expts_pos_just_1_ff_ours_training/checkpoints/epoch_39/attention_model.pt",
        "new_attention_expts_pos_just_1_ff_ours"
    ],
    
]


    for model in models:
        for file in files:
            translations  = [
                    # True, 
                    False
                ]
            for use_translations in translations:
                # languages = ["english", "russian", "french", "italian", "romanian", "turkish", "ukrainian"]
                # languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
                # languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
                languages = ["italian"]
                for language in tqdm(languages, desc="Processing Languages"):
                    print(f"Processing language: {language}")
                    print(f"translations : {use_translations}")
                    
                    inference_folder = f"input/train_infer/{language}/new_split/{file}"
                    bulk_inference = True
                    print(model[3])
                    inference = Inference(
                        inference_folder=inference_folder, 
                        bulk_inference=bulk_inference,
                        use_translations=use_translations,
                        device='cuda:1',
                        language=language,
                        question_model_name_or_path = model[0],
                        ctx_model_name_or_path = model[1],
                        attention_model = model[2],
                        comment=model[3],
                    )
                    inference.main()
                    
                    print(f"Completed processing for language: {language}")
                    print("*" * 40)
                print(f"Inference process completed for all languages for {'translated' if use_translations else 'not translated'} for model {model} for {file} datapoints")