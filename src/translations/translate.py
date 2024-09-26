import torch # type: ignore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer# type: ignore
import os
import json
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

login(os.getenv('HUGGING_FACE_KEY'))

from src.models.vector_db.commons.input_loader import InputLoader

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    
    model.to(device)
    
    return model, tokenizer


def translate_batch(queries, model, tokenizer, device, source_language):
    target_lang = "eng_Latn"
    language_mapping = {"english": "eng_Latn", "french": "fra_Latn", "italian": "ita_Latn", "romanian": "ron_Latn", "russian": "rus_Cyrl", "turkish": "tur_Latn", "ukrainian": "ukr_Cyrl"}
    
    src_lang = language_mapping[source_language]
    tokenizer.src_lang = src_lang
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=100,
        num_beams=5
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def load_input(language):
    input_data_path = os.path.join("output/dataset_outputs",language,"done")
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    input_loader = InputLoader()
    total_inference_datapoints = []
    for file in files:
        individual_datapoints = input_loader.load_data(data_file=file)
        total_inference_datapoints.extend(individual_datapoints) # type: ignore
    
    query_set = set()
    for item in total_inference_datapoints:
        for individual in item["query"]:
            query_set.add(individual)
    
    return list(query_set)

def dump_data(language, data):
    path = "output/translation_outputs"
    os.makedirs(path, exist_ok=True)  
    file_path = os.path.join(path, f"query_translations_{language}.json")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    languages = ["russian"]
    model, tokenizer = load_model(device) 
    for language in tqdm(languages):
        print(language)
        data = load_input(language)
        translations = []
        
        batch_size = 16
        for i in tqdm(range(0, len(data), batch_size)):
            batch_queries = data[i:i+batch_size]
            
            batch_translations = translate_batch(queries=batch_queries, model=model, tokenizer=tokenizer, device=device, source_language=language)
            
            for query, trans in zip(batch_queries, batch_translations):
                translations.append(
                    {
                        "original": query,
                        "translation": trans
                    }
                )
        print(len(translations))
        dump_data(language=language, data=translations)










# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer # Updated imports
# import torch # type: ignore
# import os
# import json
# from tqdm import tqdm

# from src.models.vector_db.commons.input_loader import InputLoader

# def load_model():
#     tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
#     model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B", torch_dtype=torch.float16).to("cuda").eval()
#     return model, tokenizer

# def translate(query, model, tokenizer):
#     # Update target language code as per M2M100
#     target_language = "en"
#     source_language = "fr"  # Update as needed or pass dynamically

#     tokenizer.src_lang = source_language
#     inputs = tokenizer(query, return_tensors="pt", truncation=False).to("cuda")

#     # Generating translation with M2M100 model specific parameters
#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.get_lang_id(target_language), 
#         max_length=1000,
#         num_beams=20,  # Use beam search for better translation quality
#         no_repeat_ngram_size=2  # Avoid repeating phrases
#     )
#     decoded_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
#     print("Decoded Translation:", decoded_translation)  

#     return decoded_translation[0]

# def load_input(language):
#     input_data_path = os.path.join("/srv/upadro/dataset", language, "done")
#     files = []
#     for (dirpath, dirnames, filenames) in os.walk(input_data_path):
#         for filename in filenames:
#             if "json" in filename:
#                 files.append(os.path.join(dirpath, filename))
#     input_loader = InputLoader()
#     total_inference_datapoints = []
#     for file in files:
#         individual_datapoints = input_loader.load_data(data_file=file)
#         total_inference_datapoints.extend(individual_datapoints) # type: ignore
    
#     query_set = set()
#     for item in total_inference_datapoints:
#         query_set.add(", ".join(item["query"]))
    
#     return list(query_set)

# def dump_data(language, data):
#     path = "output/translation_outputs"
#     os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
#     file_path = os.path.join(path, f"query_translations_{language}.json")
#     with open(file_path, "w", encoding="utf-8") as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
#     language_map = {
#         "french": "fr",
#         "italian": "it",
#         "romanian": "ro",
#         "russian": "ru",
#         "turkish": "tr",
#         "ukrainian": "uk"
#     }

#     for language in tqdm(languages):
#         print(language)
#         data = load_input(language)
#         translations = []
#         model, tokenizer = load_model()
#         for query in tqdm(data):
#             src_lang = language_map[language]
#             trans = translate(query=query, model=model, tokenizer=tokenizer)
#             translations.append(
#                 {
#                     "original": query,
#                     "translation": trans
#                 }
#             )
#         print(len(translations))
#         dump_data(language=language, data=translations)
