# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import random
import copy
import json
from torch.utils.data import Dataset, DataLoader, Subset
import re
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import pdb

# Load API credentials from .env file
load_dotenv()

# Initialize Azure OpenAI Chat Model
chat_model = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Define the instructions
PROMPT = "You are given a 2D text-based traffic map representing a bird’s-eye view (BEV) of a road layout with a resolution of 20×11. Each cell contains traffic-related objects separated by commas. Some of the cells are masked and are represented as <mask>. Your task is to predict the missing objects in the masked areas based on the surrounding context."
INSTRUCT1 = "Only predict the masked areas (<mask>) without modifying other parts of the map."
INSTRUCT2 = "The output should be the objects in three masked cells, where each cell is separated by '|'."
INSTRUCT3 = "e.g. The output should look like: 'lane|pedestrian crossing|ego car' without any other words"

# Combine instructions
SYSTEM_PROMPT = f"{PROMPT}\n{INSTRUCT1}\n{INSTRUCT2}\n{INSTRUCT3}"

class_list = [
    'adult',
    'animal',
    'barrier',
    'bicycle',
    'bicycle rack',
    'bus bendy',
    'bus rigid',
    'car',
    'car park area',
    'child',
    'construction vehicle',
    'construction worker',
    'debris',
    'ego car',
    'emergency police',
    'empty',
    'intersection',
    'lane',
    'motorcycle',
    'pedestrian crossing',
    'pedestrian crossing stop area',
    'personal mobility',
    'police officer',
    'pushable pullable',
    'stop sign area',
    'stroller',
    'traffic cone',
    'traffic light stop area',
    'trailer',
    'truck',
    'turn stop area',
    'walkway',
    'wheelchair'
]


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
                if area == []:
                    area = ['empty']
                if area:
                    unique_concepts = sorted(list(set(area)))
                    scene[i][j] = [convert_concatenated_words(concept) for concept in unique_concepts]
    return data


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

    def __init__(self, data_file):
        """
        Initializes the SceneDataset with the dataset and tokenizer.

        Parameters:
        - data (list): The dataset, where each item is expected to be a dictionary.
        - tokenizer (Tokenizer): The tokenizer used for encoding the text.
        - input_max_length (int): The maximum allowed length of the tokenized input sequences.
        - max_length (int): The maximum allowed length of the tokenized target sequences.
        """
        self.data_file = data_file
        self._load_data()

    def _load_data(self):
        with open(self.data_file, 'r') as file:
            self.data = json.load(file)
            self.data = remove_duplicates(self.data)
        self.ids = list(self.data.keys())

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.ids)

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
        scene_map = copy.deepcopy(self.data[self.ids[idx]]['area_embedding'])
        h = len(scene_map)
        w = len(scene_map[0])
        ## mask
        numbers = random.sample(range(h * w), 3)
        numbers.sort()
        target = []
        for i in numbers:
            target.append(",".join(scene_map[i // w][i % w]))
            scene_map[i // w][i % w] = ['<mask>']

        traffic_map = ""
        for row in scene_map:
            for cell in row:
                traffic_map = traffic_map + ",".join(cell) + '|'
            traffic_map = traffic_map + '\n'

        return {
            'traffic_map': traffic_map,
            'target': target
        }


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from openai import error
import time

precitions_all = []
target_all = []
dataset = SceneDataset(data_file='element_positions_20x11_original.json')

subset_indices = random.sample(range(len(dataset)), 100)
subset_dataset = Subset(dataset, subset_indices)

for i in tqdm(subset_dataset):
    traffic_map = i['traffic_map']
    target = i['target']
    user_message = f"Here is the traffic map:\n{traffic_map}\nPredict the masked areas."

    # Generate response
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    for i in range(10):
        try:
            response = chat_model(messages)
            break
        except error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in 5 seconds...")
            time.sleep(5)  # Wait before retrying

    # Print the output
    predictions = response.content
    predictions = predictions.split('|')[:3]
    while len(predictions) < 3:
        predictions.append("None")
    precitions_all = precitions_all + predictions
    target_all = target_all + target

class_num = len(class_list) + 1  # one for others

ground_truth = []
predictions = []
for cell in target_all:
    cell_list = cell.split(',')
    label = np.zeros(class_num)
    for obj in cell_list:
        label[class_list.index(obj)] = 1
    ground_truth.append(label)
ground_truth = np.array(ground_truth)

for cell in precitions_all:
    cell_list = cell.split(',')
    label = np.zeros(class_num)
    for obj in cell_list:
        if obj in class_list:
            label[class_list.index(obj)] = 1
        else:
            label[-1] = 1
    predictions.append(label)
predictions = np.array(predictions)

# Calculate Precision, Recall, and Accuracy for multi-label classification
precision = precision_score(ground_truth, predictions,
                            average='macro')  # Use average='micro' or 'weighted' for other types
recall = recall_score(ground_truth, predictions, average='macro')
accuracy = accuracy_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions, average="macro")

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
