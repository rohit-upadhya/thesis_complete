# import os, json, pandas as pd


# def load_files_for_lang(input_data_path):
#     files = []
#     for (dirpath, dirnames, filenames) in os.walk(input_data_path):
#         for filename in filenames:
#             if "json" in filename:
#                 files.append(os.path.join(dirpath, filename))
#     return files

# def loads_json(load_file_name):
#     with open(load_file_name, "r", encoding="utf-8") as stream:
#         data = json.load(stream)
#     return data


# if __name__=="__main__":
    
#     languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    
#     final_counts = []
#     for language in languages:
        
#         splits = ["train_test_val", "val", "test", "unique_query"]
        
#         list_query = {
#             "train_test_val": [],
#             "val": [],
#             "test": [],
#             "unique_query": []
#         }
#         train_list_query = []
#         val_list_query = []
#         test_list_query = []
#         unique_query_list_query = []
#         counts = {
#             "language": language,
#             "train_test_val": 0,
#             "val": 0,
#             "test": 0,
#             "unique_query": 0
#         }
#         train_count = 0
#         val_count = 0
#         test_count = 0
#         unique_query_count = 0
#         for split in splits:
#             folder = f"/srv/upadro/dataset/{language}/new_split/{split}"
            
#             files = load_files_for_lang(folder)
            
#             for file in files:
#                 data = loads_json(file)
                
#                 counts[split] += len(data)
#                 query_set = set()
#                 for item in data:
#                     query_set.add(
#                         f"{','.join(item['query'])} {item['link']}"
#                     )
#                 list_query['splits'].extend(
#                     list(query_set)
#                 )
#         #TODO: check overlap between the different splits

#         final_counts.append(counts)
    
#     df = pd.DataFrame(final_counts)

#     df.to_csv("output/new_split_info/dataset_numbers.csv", index=False)

#     print("Data successfully saved to output_file.csv")


import os
import json
import pandas as pd

def load_files_for_lang(input_data_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    return files

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data


if __name__ == "__main__":

    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]

    final_counts = []
    for language in languages:

        splits = ["train_test_val", "val", "test", "unique_query"]

        list_query = {
            "train_test_val": set(),
            "val": set(),
            "test": set(),
            "unique_query": set()
        }

        counts = {
            "language": language,
            "train_test_val": 0,
            "val": 0,
            "test": 0,
            "unique_query": 0
        }

        for split in splits:
            folder = f"input/train_infer/{language}/new_split/{split}"
            files = load_files_for_lang(folder)

            for file in files:
                data = loads_json(file)
                # counts[split] += len(data)
                query_set = set()
                for item in data:
                    query_set.add(
                        f"{','.join(item['query'])} {item['link']}"
                    )
                list_query[split].update(query_set)
                counts[split] += len(query_set)
        # Checking overlaps between different splits
        train_test_val_overlap_val = list_query['train_test_val'].intersection(list_query['val'])
        train_test_val_overlap_test = list_query['train_test_val'].intersection(list_query['test'])
        val_overlap_test = list_query['val'].intersection(list_query['test'])
        
        unique_overlap_train_test_val = list_query['unique_query'].intersection(list_query['train_test_val'])
        unique_overlap_val = list_query['unique_query'].intersection(list_query['val'])
        unique_overlap_test = list_query['unique_query'].intersection(list_query['test'])

        counts['train_val_overlap'] = len(train_test_val_overlap_val)
        counts['train_test_overlap'] = len(train_test_val_overlap_test)
        counts['val_test_overlap'] = len(val_overlap_test)
        counts['unique_train_test_val_overlap'] = len(unique_overlap_train_test_val)
        counts['unique_val_overlap'] = len(unique_overlap_val)
        counts['unique_test_overlap'] = len(unique_overlap_test)
        counts['total'] = counts['train_test_val'] + counts['val'] + counts['test'] + counts['unique_query']

        final_counts.append(counts)

    df = pd.DataFrame(final_counts)

    output_path = "output/new_split_info/dataset_numbers.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Data successfully saved to {output_path}")



# import os
# import json
# import pandas as pd
# from multiprocessing import Pool


# def load_files_for_lang(input_data_path):
#     """Load all JSON files from a given directory."""
#     files = []
#     for (dirpath, dirnames, filenames) in os.walk(input_data_path):
#         for filename in filenames:
#             if filename.endswith(".json"):
#                 files.append(os.path.join(dirpath, filename))
#     return files


# def loads_json(load_file_name):
#     """Load JSON content from a file."""
#     try:
#         with open(load_file_name, "r", encoding="utf-8") as stream:
#             data = json.load(stream)
#         return data
#     except (json.JSONDecodeError, FileNotFoundError) as e:
#         print(f"Error loading {load_file_name}: {e}")
#         return []


# def process_language(language, base_folder):
#     """Process a single language to calculate dataset counts and overlaps."""
#     splits = ["train_test_val", "val", "test", "unique_query"]

#     list_query = {split: set() for split in splits}
#     counts = {
#         "language": language,
#         "train_test_val": 0,
#         "val": 0,
#         "test": 0,
#         "unique_query": 0,
#     }

#     for split in splits:
#         folder = os.path.join(base_folder, language, "new_split", split)
#         files = load_files_for_lang(folder)

#         for file in files:
#             data = loads_json(file)
#             counts[split] += len(data)

#             for item in data:
#                 query_str = f"{','.join(item['query'])} {item['link']}"
#                 list_query[split].add(query_str)

#     # Compute overlaps
#     overlaps = {
#         "train_val_overlap": len(list_query["train_test_val"].intersection(list_query["val"])),
#         "train_test_overlap": len(list_query["train_test_val"].intersection(list_query["test"])),
#         "val_test_overlap": len(list_query["val"].intersection(list_query["test"])),
#         "unique_train_test_val_overlap": len(list_query["unique_query"].intersection(list_query["train_test_val"])),
#         "unique_val_overlap": len(list_query["unique_query"].intersection(list_query["val"])),
#         "unique_test_overlap": len(list_query["unique_query"].intersection(list_query["test"])),
#     }

#     counts.update(overlaps)
#     counts["total"] = sum(counts[split] for split in splits)

#     # Save unique query counts
#     counts["unique_query_counts"] = {
#         split: len(list_query[split]) for split in splits
#     }

#     return counts


# def main():
#     languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
#     base_folder = "input/train_infer"
#     output_path = "output/new_split_info/dataset_numbers.json"

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Use multiprocessing to process languages in parallel
#     with Pool() as pool:
#         results = pool.starmap(process_language, [(lang, base_folder) for lang in languages])

#     # Save results to a JSON file
#     with open(output_path, "w", encoding="utf-8") as json_file:
#         json.dump(results, json_file, ensure_ascii=False, indent=4)

#     print(f"Data successfully saved to {output_path}")


# if __name__ == "__main__":
#     main()
