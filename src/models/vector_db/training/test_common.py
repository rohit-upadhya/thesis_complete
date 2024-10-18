from src.models.single_datapoints.common.data_loader import InputLoader
input_loader = InputLoader()
import os


def load_all_input_from_dir(input_data_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        total_inference_datapoints = []
        for file in files:
            individual_datapoints = input_loader.load_data(data_file=file)
            total_inference_datapoints.extend(individual_datapoints) # type: ignore
        return total_inference_datapoints

data = load_all_input_from_dir('src/models/single_datapoints/common')

counts = 0
for points in data:
    counts += len(points["paragraph_numbers"])

print(counts)