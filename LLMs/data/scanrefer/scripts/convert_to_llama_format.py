import json
import random

def convert_to_jsonl(input_qa_file, input_caption_file, output_file, is_question_not_scene=False, subset=1.0, is_train=False):
    query_key_name = "question_id" if is_question_not_scene else "scene_id"
    extension = "train" if is_train else "val"
    
    # Load data from "scanrefer/ScanRefer_cluster_val.json"
    with open(input_caption_file, "r") as f:
        scanrefer_data = json.load(f)
    # Convert the list to a dictionary for efficient lookup by scene_id
    scanrefer_dict = {d[query_key_name]: d['description'] for d in scanrefer_data}

    # Load data from "qa/SQA_balanced_val.json"
    with open(input_qa_file, "r") as f:
        sqa_data = json.load(f)

    # Open "sqa3d/val.jsonl" for writing
    total_n = len(sqa_data)
    subset_n = int(total_n * subset)
    if subset < 1.0:
        output_file = output_file.split('.jsonl')[0] + '_subset.jsonl'
    with open(output_file, "w") as outfile:
        subset_indices = random.sample(range(total_n), subset_n)
        # subset_indices = range(total_n)[:200]
        subset = [sqa_data[i] for i in subset_indices]
        for item in subset:
            query_id = item[query_key_name]
            # If the scene_id exists in the scanrefer data, add the description
            if query_id in scanrefer_dict:
                item['description'] = scanrefer_dict[query_id]
                # Write the dictionary to the jsonl file, one per line
                json.dump(item, outfile)
                outfile.write('\n')
    print("Writent to {}".format(output_file))
    
if __name__ == '__main__':
    # question_or_scene # subset_or_full_set
    input_qa_file = "qa/SQA_balanced_val.json"
    input_caption_file = 'scanrefer/ScanRefer_wcds15_val.json'
    output_file = '/fsx-llm/shared/eval/datasets/sqa3d/subset_mini/sqa3d/val_wcds15.jsonl'
    convert_to_jsonl(input_qa_file, input_caption_file, output_file, is_question_not_scene=False, subset=0.05)
