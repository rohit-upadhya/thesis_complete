import json
import os
import pandas as pd


if __name__ == "__main__":
    base_path = "/srv/upadro/data_analysis/unseen_queries"
    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    splits = ["unique_query_test", "train_test_val", "train", "test", "val"]


    data = {lang: {} for lang in languages}

    for lang in languages:
        for split in splits:
            file_path = os.path.join(base_path, split, "counts", f"{lang}.json")
            try:
                with open(file_path, 'r') as f:
                    stats = json.load(f)
                    data[lang][f"{split}_pairs"] = stats["number_of_q_d_pairs"]
                    data[lang][f"{split}_queries"] = stats["number_of_unique_queries"]
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                data[lang][f"{split}_pairs"] = "N/A"
                data[lang][f"{split}_queries"] = "N/A"

    df = pd.DataFrame.from_dict(data, orient='index')

    column_order = ['unique_query_test_pairs', 'unique_query_test_queries',
                    'train_test_val_pairs', 'train_test_val_queries',
                    'train_pairs', 'test_pairs', 'val_pairs']
    df = df[column_order]

    df.columns = pd.MultiIndex.from_tuples([
        ('unique_query_test', 'QDR Triplets'),
        ('unique_query_test', 'Queries'),
        ('train_test_val', 'QDR Triplets'),
        ('train_test_val', 'Queries'),
        ('train', 'QDR Triplets'),
        ('test', 'QDR Triplets'),
        ('val', 'QDR Triplets')
    ])

    df.insert(0, "Language", df.index)
    df = df.reset_index(drop=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_colwidth', 20)

    print(df.to_string(index=False))

    df.to_csv("dataset_stats.csv")

