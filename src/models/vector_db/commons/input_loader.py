from dataclasses import dataclass
from typing import Text, Dict
import os
import yaml
import json

class InputLoader:
    def load_data(self, data_file: Text):
        _, file_extension = os.path.splitext(data_file)
        few_shot = {}
        if "json" in file_extension.lower():
            few_shot = self._load_json(data_file)
        if "yaml" in file_extension.lower() or "yml" in file_extension.lower():
            few_shot = self._load_yaml(data_file)
        return few_shot
        
    def load_config(self, config_file: Text):
        _, file_extension = os.path.splitext(config_file)
        config = {}
        if "json" in file_extension.lower():
            config = self._load_json(config_file)
        if "yaml" in file_extension.lower() or "yml" in file_extension.lower():
            config = self._load_yaml(config_file)
        return config
    
    def _load_yaml(self, yaml_file:Text):
        with open(yaml_file, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")
                return None
    
    def _load_json(self, json_file:Text):
        with open(json_file, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as exc:
                print(f"Error loading JSON file: {exc}")
                return None

