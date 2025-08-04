# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import sys
import os
import shutil
import warnings
import torch
import transformers
from torch import optim
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config, get_linear_schedule_with_warmup
import json
import random
import copy
import yaml
from sklearn.model_selection import train_test_split
import re
import argparse


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import train_test_split



# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Process the config file.')
parser.add_argument('config_path', type=str, help='Path to the config.yaml file')
args = parser.parse_args()

# Function to read config values
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration from the provided paths
config = read_config(args.config_path)



#config
epochs = config['training']['epochs']
lr = float(config['training']['learning_rate'])
batch_size = config['training']['batch_size']
warmup_steps = config['training']['warmup_steps']
max_length = config['training']['max_length']
input_max_length = config['training']['input_max_length']
seed = config['training']['seed']
train_size = config['training']['train_size']
orientation_bool = config['data']['orientation']
distance_bool = config['data']['distance']
country_bool = config['data']['country']
# augmentation_multiplier = config['training']['augmentation']


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(seed)

train_first_scene_mask_number = config['scene_masking']['train_first_scene_mask_number']
train_second_scene_mask_number = config['scene_masking']['train_second_scene_mask_number']
val_first_scene_mask_number = config['scene_masking']['val_first_scene_mask_number']
val_second_scene_mask_number = config['scene_masking']['val_second_scene_mask_number']
test_first_scene_mask_number = config['scene_masking']['test_first_scene_mask_number']
test_second_scene_mask_number = config['scene_masking']['test_second_scene_mask_number']
train_size = 0.8

next_scene_bool = config['scene_masking']['next_scene']
outer_rows = config['scene_masking']['outer_rows']
outer_cols = config['scene_masking']['outer_cols']

if not next_scene_bool:
    model_save_path = config['paths']['model_save_path'] + '_multi_mask_prediction' + str(train_first_scene_mask_number) + '_' + str(train_second_scene_mask_number)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    log_path = model_save_path + '/log/' + model_save_path + '.json'

    log_path_without_json = model_save_path + '/log'
    if not os.path.exists(log_path_without_json):
        os.makedirs(log_path_without_json)

    best_save_path = model_save_path +'/best_val_loss'
    if not os.path.exists(best_save_path):
        os.makedirs(best_save_path)
else:
    model_save_path = config['paths']['model_save_path'] + '_next_scene_prediction'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    log_path = model_save_path + '/log/' + model_save_path + '.json'

    log_path_without_json = model_save_path + '/log'
    if not os.path.exists(log_path_without_json):
        os.makedirs(log_path_without_json)

    best_save_path = model_save_path +'/best_val_loss'
    if not os.path.exists(best_save_path):
        os.makedirs(best_save_path)

tokenizer_path = config['paths']['tokenizer_path']
model_load_path = config['paths']['model_load_path']

with open(config['paths']['matrix_file_path']) as f:
    data = json.load(f)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
config = T5Config(n_positions = 4096)
model = T5ForConditionalGeneration(config).from_pretrained(model_load_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)

from collections import Counter
import matplotlib.pyplot as plt

def mask_random_areas(scene, mask_number, start_mask_id=0):
    """
    Randomly masks specified number of areas in a given scene.

    This function selects non-empty areas in the scene and masks them with unique identifiers.
    It updates the scene in-place and returns details about the masked areas, including
    the updated start_mask_id, concept counter, and area concept counts.

    Parameters:
    - scene (list): A 2D list representing the scene to be masked.
    - mask_number (int): The number of areas to mask.
    - start_mask_id (int, optional): The starting ID to use for generating mask identifiers.

    Returns:
    - tuple: Contains the list of masked areas, the next start mask ID, concept counter,
             and area concept counts.
    """
    non_empty_indices = [(i, j) for i, row in enumerate(scene) for j, area in enumerate(row) if area]

    if non_empty_indices:
        masked_areas = []
        mask_indices = []
        concept_counter = Counter()  
        area_concept_counts = Counter()

        mask_number = min(mask_number, len(non_empty_indices))
        mask_indices = random.sample(non_empty_indices, mask_number)
        mask_indices.sort()

        for mask_id, (i, j) in enumerate(mask_indices):
            masked_area = ','.join(scene[i][j])
            masked_areas.append(masked_area)

            concepts = scene[i][j]
            for concept in concepts:
                concept_counter.update(concept.split(','))
            
            area_concept_counts[len(concepts)] += 1
            scene[i][j] = ["<extra_id_"+str(start_mask_id + mask_id)+">"]
            
        return masked_areas, start_mask_id + mask_number, concept_counter, area_concept_counts
    else:
        return None, start_mask_id, Counter(), Counter()
    

def mask_scene(scene, outer_rows, outer_cols, start_mask_id=0):
    """
    Masks the outer rows and columns of a scene and uniquely identifies inner areas.

    This function masks specified outer rows and columns of a scene with a placeholder 
    and assigns unique identifiers to the inner unmasked areas. The function is typically 
    used to preprocess the scene data before feeding it into a model.

    Parameters:
    - scene (list): A 2D list representing the scene.
    - outer_rows (int): Number of outer rows to mask.
    - outer_cols (int): Number of outer columns to mask.
    - start_mask_id (int, optional): Initial ID to use for the inner masked areas.

    Returns:
    - list: A list of the inner areas that were masked.
    """
    # Mask the outer rows and columns with <unk>
    for i in range(len(scene)):
        for j in range(len(scene[i])):
            if i < outer_rows or i >= len(scene) - outer_rows:
                scene[i][j] = ["<unk>"]
            elif j < outer_cols or j >= len(scene[0]) - outer_cols:
                scene[i][j] = ["<unk>"]
    # Find all inner indices for masking with unique IDs
    inner_indices = [(i, j) for i in range(outer_rows, len(scene) - outer_rows)
                     for j in range(outer_cols, len(scene[0]) - outer_cols)]
    # Mask the inner areas and keep track of the masked areas
    masked_areas = []
    for mask_id, (i, j) in enumerate(inner_indices, start=start_mask_id):
        if scene[i][j] != "<unk>":  # Only mask if it's not already <unk>
            masked_area = ','.join(scene[i][j])
            if not scene[i][j]:
                masked_areas.append("<empty>")
            else:
                masked_areas.append(masked_area)
            scene[i][j] = [f"<extra_id_{mask_id}>"]

    # Return the list of masked areas and the next starting mask ID
    return masked_areas
    
def check_more_than_half_empty(scene):
    """
    Checks if more than half of the areas in the scene are empty.

    This function calculates the total number of areas in the scene and determines 
    if the number of empty areas exceeds half of the total areas.

    Parameters:
    - scene (list): A 2D list representing the scene.

    Returns:
    - bool: True if more than half of the areas are empty, False otherwise.
    """
    total_areas = len(scene) * len(scene[0])
    non_empty_indices = [(i, j) for i, row in enumerate(scene) for j, area in enumerate(row) if area]
    empty_areas = total_areas - len(non_empty_indices)

    if empty_areas > total_areas // 2:
        return True
    else:
        return False

def scene_to_str(scene):
    """
    Converts a scene representation into a string format.

    This function transforms a 2D list representing a scene into a single string, 
    using specific separators to denote rows and columns.

    Parameters:
    - scene (list): A 2D list representing the scene.

    Returns:
    - str: A string representation of the scene.
    """
    return ' <row_sep> '.join(' <col_sep> '.join('<empty>' if not area else ' <concept_sep> '.join(area) for area in row) for row in scene)

def get_next_scene(data, current_scene_id):
    """
    Retrieves the next scene and its ID based on the current scene ID.

    Parameters:
    - data (dict): A dictionary containing scene data.
    - current_scene_id (str): The ID of the current scene.

    Returns:
    - tuple: The next scene and its ID.
    """
    next_scene_id = data[current_scene_id]['next_scene_ID']
    next_scene = data[next_scene_id]['area_embedding'] if next_scene_id in data else None
    return next_scene, next_scene_id

def convert_concatenated_words(input_string):
    """
    Converts concatenated words into a space-separated format.

    This function identifies concatenated words where a lowercase letter is followed 
    by an uppercase letter, inserts a space, and ensures all characters are lowercase. 
    Specific adjustments are made for known edge cases.

    Parameters:
    - input_string (str): The string containing concatenated words.

    Returns:
    - str: The processed string with spaces inserted between concatenated words.
    """
    # Use regular expressions to insert spaces before capital letters
    output_string = re.sub(r'([a-z])([A-Z])', r'\1 \2', input_string)
    # Lowercase the entire string
    output_string = output_string.lower()
    # Handle 'Ped...' and 'Carpark...' edge cases
    output_string = output_string.replace('ped', 'pedestrian').replace('carpark', 'car park')
    return output_string

def remove_duplicates(data):
    """
    Removes duplicate concepts within each area of every scene.

    Iterates over each scene and its areas, ensuring each concept is unique within its area. 
    Also processes the concepts to a normalized form.

    Parameters:
    - data (dict): A dictionary containing scenes and their area embeddings.

    Returns:
    - dict: The updated data with duplicates removed from each scene's areas.
    """
    for scene_id in data:
        scene = data[scene_id]['area_embedding']
        for i, row in enumerate(scene):
            for j, area in enumerate(row):
                if area:
                    unique_concepts = sorted(list(set(area)))
                    scene[i][j] = [convert_concatenated_words(concept) for concept in unique_concepts]
    return data

def preprocess_data(data, first_scene_mask_number, second_scene_mask_number, orientation_bool, distance_bool, country_bool, oversample_concepts=[], oversample_multiplier=1):
    """
    Preprocesses the scene data by applying various transformations and masking.

    This function processes the given data by removing duplicates, applying random masking, 
    and preparing the data format suitable for model training, considering various flags like 
    orientation, distance, and country specifics.

    Parameters:
    - data (dict): The raw scene data to preprocess.
    - first_scene_mask_number (int): The number of areas to mask in the first scene.
    - second_scene_mask_number (int): The number of areas to mask in the second scene.
    - orientation_bool (bool): Flag to include orientation data.
    - distance_bool (bool): Flag to include distance data.
    - country_bool (bool): Flag to include country data.
    - oversample_concepts (list, optional): Concepts to oversample.
    - oversample_multiplier (int, optional): Multiplier for oversampling.

    Returns:
    - tuple: Preprocessed data along with concept counters and area concept counts.
    """
    data = remove_duplicates(data)
    preprocessed_data = []
    concept_counter = Counter()
    total_concept_counter = Counter()
    area_concept_counts = Counter()
    c = 0
    for scene_id in data:
        scene = copy.deepcopy(data[scene_id]['area_embedding'])
        next_scene, next_scene_id = copy.deepcopy(get_next_scene(data, scene_id))
        empty = False #check_more_than_half_empty(scene)

        if next_scene and not empty:
            for row in next_scene:
                for area in row:
                    if area:
                        for concept in area:
                            total_concept_counter.update(concept.split(','))

            masked_areas, last_mask_id, concept_counter_scene, area_concept_counts_scene = mask_random_areas(scene, first_scene_mask_number)
            next_masked_areas, _, concept_counter_next_scene, area_concept_counts_next_scene = mask_random_areas(next_scene, second_scene_mask_number, last_mask_id)

            concept_counter += concept_counter_scene + concept_counter_next_scene
            area_concept_counts += area_concept_counts_scene + area_concept_counts_next_scene
            
            if (masked_areas and next_masked_areas) or (first_scene_mask_number == 0 and next_masked_areas) or (second_scene_mask_number == 0 and masked_areas):
                next_scene_str = scene_to_str(next_scene)
                scene_str = scene_to_str(scene)
                scene_distance = str(data[scene_id]['distance_between_scenes'])
                scene_orientation = str(data[scene_id]['orientation_difference'])
                sequence_ID = data[scene_id]['sequence_ID']
                country = data[scene_id]['country']
                combined_scene_str = " <scene_start> " + scene_str + " <scene_sep> " + next_scene_str
                if orientation_bool:
                    combined_scene_str = " <orientatoin_diff> " + scene_orientation + combined_scene_str
                if distance_bool:
                    combined_scene_str = " <dist> " + scene_distance + combined_scene_str
                if country_bool:
                    combined_scene_str = " <country> " + country + combined_scene_str
                target_text = " ".join(f"<extra_id_{i}> {area.replace(',',' <concept_sep> ')}" for i, area in enumerate(masked_areas + next_masked_areas))
                target_text = target_text + ' <extra_id_' + str(len(masked_areas + next_masked_areas)) + '>'
                
                data_point = {
                    "input_text": combined_scene_str,
                    "output_text": target_text,
                    "scene_ID": scene_id,
                    "next_scene_ID": next_scene_id,
                    "sequence_ID": sequence_ID
                }
                
#                 # Check if any of the concepts in masked areas intersect with oversample_concepts
#                 intersecting_concepts = [concept for area in masked_areas + next_masked_areas for concept in area.split(",") if concept in oversample_concepts]
                
#                 # If there's an intersection and oversample_concepts is provided, append the data point multiple times
#                 if intersecting_concepts:
#                     for _ in range(oversample_multiplier):
#                         preprocessed_data.append(data_point)
                        
#                         # Update the counters for each oversampled data point
#                         concept_counter += concept_counter_scene + concept_counter_next_scene
#                         area_concept_counts += area_concept_counts_scene + area_concept_counts_next_scene
#                 else:
                preprocessed_data.append(data_point)
                

    return preprocessed_data, concept_counter, total_concept_counter, area_concept_counts




# train_data, val_data = train_test_split(preprocessed_data, test_size=1-train_size, random_state=seed)
# val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=seed)

#val_preprocessed_data, val_concept_counter, val_total_concept_counter, val_area_concept_counts = preprocess_data(val_data, val_first_scene_mask_number, val_second_scene_mask_number)
#test_preprocessed_data, test_concept_counter, test_total_concept_counter, test_area_concept_counts = preprocess_data(test_data, test_first_scene_mask_number, test_second_scene_mask_number)
#train_data, val_data = train_test_split(preprocessed_data, test_size=1-train_size, random_state=seed)
#val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=seed)

def random_remask_scenes(train_data, first_scene_mask_number, second_scene_mask_number):
    """
    Randomly remasks scenes in the training data based on specified mask numbers.

    This function iterates over the training data, remasking the scenes with new random masks. 
    This can be used for data augmentation or to introduce variability in the training process.

    Parameters:
    - train_data (list): The training data containing scenes to be remasked.
    - first_scene_mask_number (int): Number of areas to mask in the first scene of each pair.
    - second_scene_mask_number (int): Number of areas to mask in the second scene of each pair.

    Returns:
    - None: The function operates in-place and modifies the train_data directly.
    """
    for i in range(len(train_data)):
        data = train_data[i]
        
        scene_start_index = data['input_text'].find('<scene_start>')

        if scene_start_index != -1:
            scene_text = data['input_text'][scene_start_index + len('<scene_start>'): ]

        extra_id_mappings = {}
        for match in re.findall(r'<extra_id_(\d+)> ([^<]*)', data['output_text']):
            extra_id, replacement_text = match
            extra_id_mappings[f'<extra_id_{extra_id}>'] = replacement_text.rstrip()

        for extra_id, replacement_text in extra_id_mappings.items():
             scene_text = scene_text.replace(extra_id, replacement_text)
        
        
        scenes = scene_text.split('<scene_sep>')
        output_text = ''
        extra_id_counter = 0

        mask_numbers = [first_scene_mask_number, second_scene_mask_number]

        for scene_idx, scene in enumerate(scenes):
            rows = scene.split('<row_sep>')
            masked_scene_rows = []

            # Find all non-empty areas
            non_empty_areas = []
            for row_idx, row in enumerate(rows):
                areas = row.split('<col_sep>')
                for area_idx, area in enumerate(areas):
                    if area != ' <empty> ' and area != ' <empty>' and area != '<empty> ':
                        non_empty_areas.append((row_idx, area_idx))

            # Randomly pick areas to be masked
            random.shuffle(non_empty_areas)

            mask_count = min(len(non_empty_areas), mask_numbers[scene_idx])

            masked_indices = set(non_empty_areas[:mask_count])

            for row_idx, row in enumerate(rows):
                areas = row.split('<col_sep>')
                masked_row = []

                for area_idx, area in enumerate(areas):
                    if (row_idx, area_idx) in masked_indices:
                        output_text += f"<extra_id_{extra_id_counter}>{area}"
                        masked_row.append(f" <extra_id_{extra_id_counter}> ")
                        extra_id_counter += 1
                    else:
                        masked_row.append(area)

                masked_scene_rows.append('<col_sep>'.join(masked_row))

            scenes[scene_idx] = '<row_sep>'.join(masked_scene_rows)

        masked_scene_text = '<scene_sep>'.join(scenes)
        prefix_text = data['input_text'][:scene_start_index + len('<scene_start>')]
        train_data[i]['input_text'] = prefix_text + masked_scene_text
        output_text = output_text + f"<extra_id_{extra_id_counter}>"
        train_data[i]['output_text'] = output_text.rstrip()
    
    return

def unmask_input(input_text, output_text):
    """
    Replaces masked tokens in the input text with their corresponding values from the output text.

    This function iterates through the output text, identifies the masked tokens and their 
    replacements, and substitutes the original masked tokens in the input text with 
    their corresponding values.

    Parameters:
    - input_text (str): The text containing masked tokens.
    - output_text (str): The text containing replacements for the masked tokens.

    Returns:
    - str: The input text with masked tokens replaced by their corresponding values from the output text.
    """
    # Split the output_text into a list of target elements
    target_elements = output_text.split('<extra_id_')[1:]
    # Replace each <extra_id_i> token in input_text with its corresponding target element
    for target_element in target_elements:
        # Extract the index from the target element
        index = target_element.split('>')[0]
        input_text = input_text.replace(f'<extra_id_{index}>', target_element[len(index)+1:], 1)  # Replace only the first occurrence

    return input_text

def mask_ns(data, outer_rows, outer_cols):
    """
    Applies masking to the next scenes in the dataset.

    Iterates over the dataset, unmasking and then re-masking each next scene (ns) using the provided
    outer row and column specifications. This function is used to preprocess the dataset by
    applying a consistent masking strategy to all next scenes.

    Parameters:
    - data (list): The dataset containing scenes to be masked.
    - outer_rows (int): The number of rows at the border of the scene to mask.
    - outer_cols (int): The number of columns at the border of the scene to mask.

    Returns:
    - None: The function operates in-place and modifies the 'data' directly.
    """
    for i, dp in enumerate(data):
        input_text = unmask_input(dp['input_text'], dp['output_text'])
        s = input_text.split(' <scene_sep> ')[0]
        ns = input_text.split(' <scene_sep> ')[1]
        l = ns.split(' <row_sep> ')
        for i, row in enumerate(l):
            l[i] = row.split(' <col_sep> ')
            for j, col in enumerate(l[i]):
                l[i][j] = [l[i][j]]
        next_masked_areas = mask_scene(l, outer_rows=outer_rows, outer_cols=outer_cols)
        ns = scene_to_str(l)
        target_text = " ".join(f"<extra_id_{i}> {area.replace(',',' <concept_sep> ')}" for i, area in enumerate(next_masked_areas))
        target_text = target_text + ' <extra_id_' + str(len(next_masked_areas)) + '>'
        dp['input_text'] = s + ' <scene_sep> ' + ns
        dp['output_text'] = target_text
        data[i] = dp
    return



class SceneDataset(Dataset):
    """
    A PyTorch Dataset class for handling scene data, specifically formatted for language models.

    The class encodes text data into a format suitable for training with PyTorch, including tokenization
    and converting to tensor format. It's specifically designed to handle scene descriptions and their
    corresponding tokenized outputs.

    Attributes:
    - data (list): A list of data points, where each data point is expected to be a dictionary with
                   keys 'input_text' and 'output_text'.
    - tokenizer (Tokenizer): A tokenizer object used for converting text to a sequence of integers.
    - max_length (int): The maximum length for the tokenized target sequences.
    - input_max_length (int): The maximum length for the tokenized input sequences.
    """
    def __init__(self, data, tokenizer, input_max_length=input_max_length, max_length=max_length):
        """
        Initializes the SceneDataset with the dataset and tokenizer.

        Parameters:
        - data (list): The dataset, where each item is expected to be a dictionary.
        - tokenizer (Tokenizer): The tokenizer used for encoding the text.
        - input_max_length (int): The maximum allowed length of the tokenized input sequences.
        - max_length (int): The maximum allowed length of the tokenized target sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_max_length=input_max_length
        
    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the indexed item from the dataset.

        Encodes the input and output texts at the specified index into tensors, which can be
        fed directly into a PyTorch model.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        - dict: Contains the encoded 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        item = self.data[idx]
        encoding = self.tokenizer(item['input_text'], truncation=True, max_length=self.input_max_length, padding='max_length')
        target = self.tokenizer(item['output_text'], truncation=True, max_length=self.max_length, padding='max_length')
        decode_input_text = self.tokenizer.decode(encoding['input_ids'], skip_special_tokens=False)
        decode_target_text = self.tokenizer.decode(target['input_ids'], skip_special_tokens=False)
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(target['input_ids'], dtype=torch.long)
        }


# def augmente(train_data, augmentation_multiplier, train_first_scene_mask_number, train_second_scene_mask_number):
#     original_len = len(train_data)
#     for i in range(augmentation_multiplier - 1):
#         start_idx = len(train_data) - original_len  # Get the starting index for the latest data
#         train_data_i = copy.deepcopy(train_data[start_idx:])  # Copy only the latest data
#         random_remask_scenes(train_data_i, train_first_scene_mask_number, train_second_scene_mask_number)  # Augment only the latest data
#         train_data.extend(train_data_i)  # Add it to the original list
#     return train_data    

#augmente(train_data, augmentation_multiplier, train_first_scene_mask_number, train_second_scene_mask_number)


def calculate_accuracy(preds_text, targets_text):
    """
    Calculates the accuracy of predicted texts against target texts.

    The function compares each set of predicted words (from the predicted text) against the
    corresponding set of target words, counting matches. The accuracy is the proportion of
    correctly predicted sets to the total sets.

    Parameters:
    - preds_text (str): The concatenated predicted texts, separated by specific ID tokens.
    - targets_text (str): The concatenated target texts, separated by the same ID tokens as the predictions.

    Returns:
    - float: The calculated accuracy, ranging from 0 to 1.
    """
    # Split the text by <extra_id_\d+> tags
    preds_areas = re.split(r'<extra_id_\d+>', preds_text)[1:]  # exclude the first empty string
    targets_areas = re.split(r'<extra_id_\d+>', targets_text)[1:]  # exclude the first empty string
    if preds_areas and preds_areas[-1] == "":
        preds_areas = preds_areas[:-1]
    if preds_areas and preds_areas[-1] == ' ':
        preds_areas = preds_areas[:-1]
    if targets_areas and targets_areas[-1] == "":
        targets_areas = targets_areas[:-1]     
    if targets_areas and targets_areas[-1] == ' ':
        targets_areas = targets_areas[:-1]
    
    # Initialize counters
    total_sets = len(targets_areas)
    matched_sets = 0
    
    # Iterate over each set of targets areas
    for i, targets_area in enumerate(targets_areas):
        # If the index is within the length of preds_areas, compare the sets
        if i < len(preds_areas):
            # Create sets of the words within each area
            preds_set = {word.strip() for word in preds_areas[i].split('<concept_sep>')}
            targets_set = {word.strip() for word in targets_area.split('<concept_sep>')}
            # If the sets match, increment the matched counter
            if preds_set == targets_set:
                matched_sets += 1

    # Return the ratio of matched sets to all sets
    return matched_sets / total_sets if total_sets > 0 else 0

# Data processing
preprocessed_data, train_concept_counter, train_total_concept_counter, train_area_concept_counts = preprocess_data(data, train_first_scene_mask_number, train_second_scene_mask_number, orientation_bool, distance_bool, country_bool)

data_by_sequence = {}
for data_point in preprocessed_data:
    if data_point['sequence_ID'] not in data_by_sequence:
        data_by_sequence[data_point['sequence_ID']] = []
    else:
        data_by_sequence[data_point['sequence_ID']].append(data_point)

sequence_ids = list(data_by_sequence.keys())
random.shuffle(sequence_ids)
val_size = (1 - train_size)/2

total_sequences = len(sequence_ids)
train_size = int(train_size * total_sequences)
val_size = int(val_size * total_sequences)

train_sequence_ids = sequence_ids[:train_size]
val_sequence_ids = sequence_ids[train_size:train_size + val_size]
test_sequence_ids = sequence_ids[train_size + val_size:]

train_data = []
val_data = []
test_data = []

for sequence_id in train_sequence_ids:
    train_data.extend(data_by_sequence[sequence_id])

for sequence_id in val_sequence_ids:
    val_data.extend(data_by_sequence[sequence_id])

for sequence_id in test_sequence_ids:
    test_data.extend(data_by_sequence[sequence_id])

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# next scene or multimask masking
if next_scene_bool:
    mask_ns(train_data, outer_rows, outer_cols)
    mask_ns(val_data, outer_rows, outer_cols)
    mask_ns(test_data, outer_rows, outer_cols)
if not next_scene_bool:
    random_remask_scenes(test_data, test_first_scene_mask_number, test_second_scene_mask_number)

train_dataset = SceneDataset(train_data, tokenizer)
val_dataset = SceneDataset(val_data, tokenizer)
test_dataset = SceneDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs*len(train_loader))

#gradient_accumulation_steps = 4
#num_training_steps = len(train_loader) // gradient_accumulation_steps

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
min_val_loss = 10

for epoch in range(epochs):
    total = 0
    correct = 0
    train_loss = 0
    model.train()
    
    
    for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # print(input_ids[0])
        # print(len(attention_mask[0]))
        # preds_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in labels]
        # print(preds_text)
        # print(len(labels[0]))
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        # preds = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10000)
        # preds_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in preds]
        # preds_text = [text.replace('<pad>', '').replace('</s>', '') for text in preds_text]
        # print('\n preds_text: ',preds_text)
        # targets_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in labels]
        # targets_text = [text.replace('<pad>', '').replace('</s>', '') for text in targets_text]
        # print('\n targets_text: ',targets_text)
        # for ids in preds:
        #     print('\n targets_ids_length: ', len(ids))
        # for i in range(len(targets_text)):
        #     correct += calculate_accuracy(preds_text[i], targets_text[i])
        # print('\n correct: ',correct)
        # total += len(targets_text)
        # print('\n total: ', total)
        
        #print(f'Epoch: {epoch}, Loss: {train_loss/total}, Accuracy: {correct/total}')
    train_loss /= len(train_loader)    
    train_losses.append(train_loss)
    #train_accuracies.append(correct/total)

    # Validation
    total = 0
    correct = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10000)
            preds_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in preds]
            preds_text = [text.replace('<pad>', '').replace('</s>', '') for text in preds_text]
            targets_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in labels]
            targets_text = [text.replace('<pad>', '').replace('</s>', '') for text in targets_text]

            for i in range(len(targets_text)):
                correct += calculate_accuracy(preds_text[i], targets_text[i])
            total += len(targets_text)
            val_loss += loss.item()
            print(f'Validation Loss: {val_loss/total}, Validation Accuracy: {correct/total}')
            
    if val_loss/total < min_val_loss:
        min_val_loss = val_loss/total
        model.save_pretrained(best_save_path)
    val_losses.append(val_loss/total)
    val_accuracies.append(correct/total)
    
    #randomly remasking train_data
    #random_remask_scenes(train_data, train_first_scene_mask_number, train_second_scene_mask_number)
    train_dataset = SceneDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Testing
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        preds = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10000)
        preds_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in preds]
        preds_text = [text.replace('<pad>', '').replace('</s>', '') for text in preds_text]
        print('\n test_preds_text: ',preds_text)
        targets_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in labels]
        targets_text = [text.replace('<pad>', '').replace('</s>', '') for text in targets_text]
        print('\n test_targets_text: ',targets_text)
        for i in range(len(targets_text)):
            correct += calculate_accuracy(preds_text[i], targets_text[i])
        total += len(targets_text)
        print(f'Test Accuracy: {correct/total}')

test_accuracy = correct/total

metrics = {
    'train_losses': train_losses,
    #'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'test_accuracy': test_accuracy
}

with open(log_path, 'w') as json_file:
    json.dump(metrics, json_file)

model.save_pretrained(model_save_path)
