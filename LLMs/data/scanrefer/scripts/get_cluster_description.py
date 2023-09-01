import json
import tqdm
from collections import defaultdict

# Path to the input JSON file
input_file_path = 'ScanRefer_filtered_train.json'

# Path to the output JSON file
output_file_path = 'ScanRefer_cluster_train.jsonn'

# Read the input JSON file
with open(input_file_path, 'r') as f:
    data = json.load(f)

# Use a defaultdict to collect descriptions by scene_id
scene_descriptions = defaultdict(str)

for entry in tqdm.tqdm(data, total=len(data)):
    scene_id = entry['scene_id']
    description = entry['description']
    # Concatenate descriptions for the same scene_id
    scene_descriptions[scene_id] += description + ' '

# Convert the defaultdict to a list of dicts
output_data = [{'scene_id': k, 'description': v.rstrip()} for k, v in scene_descriptions.items()]
print(len(output_data))
# Write the output JSON file
with open(output_file_path, 'w') as f:
    json.dump(output_data, f)
