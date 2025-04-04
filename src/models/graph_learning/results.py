import json
import os

def calculate_averages_from_json(file_path):
    """
    Reads a JSON file containing recall data, calculates the averages
    for recall at 2%, 5%, and 10%, and rounds them to 4 decimal places.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary with rounded averages for recall at 2%, 5%, and 10%.
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract recall values
        recall_at_2 = [lang["mean_recall_at_2_percentage"] for lang in data.values() if "mean_recall_at_2_percentage" in lang]
        recall_at_5 = [lang["mean_recall_at_5_percentage"] for lang in data.values() if "mean_recall_at_5_percentage" in lang]
        recall_at_10 = [lang["mean_recall_at_10_percentage"] for lang in data.values() if "mean_recall_at_10_percentage" in lang]

        # Calculate averages
        average_recall_at_2 = round(sum(recall_at_2) / len(recall_at_2), 4)
        average_recall_at_5 = round(sum(recall_at_5) / len(recall_at_5), 4)
        average_recall_at_10 = round(sum(recall_at_10) / len(recall_at_10), 4)

        # Return the results as a dictionary
        return {
            "average_recall_at_2": average_recall_at_2,
            "average_recall_at_5": average_recall_at_5,
            "average_recall_at_10": average_recall_at_10
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Example usage:
# result = calculate_averages_from_json('path_to_your_file.json')
# print(result)
def extract_epoch_from_filename(filename):
    """
    Extracts the epoch number from a given file path.

    Args:
        filename (str): The file path containing the epoch information.

    Returns:
        int: The extracted epoch number.
    """
    try:
        # Use split to isolate parts of the path
        parts = filename.split('epoch_')
        if len(parts) > 1:
            # Extract the number after 'epoch_'
            epoch_part = parts[1].split('_')[0]
            # print(epoch_part)
            return epoch_part
    except Exception as e:
        print(f"An error occurred while extracting the epoch: {e}")
    return None

if __name__ == "__main__":
    input_data_path = "output/inference_outputs/new_splits/new_experiments/ensemble/new_gat_ablation_last_mixing_next_5_topic_threshold_english/base/test"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    
    for file in files:
        average = calculate_averages_from_json(file)
        epoch = extract_epoch_from_filename(file)
        print(f"epoch_{epoch} : {average}")
    
    