import json
import random
import tqdm
import nltk
import spacy
from pattern.en import singularize
import tqdm
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def extract_nouns(text, nlp):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence
    words = nltk.word_tokenize(text)

    # Get the parts of speech of each word
    pos_tags = nltk.pos_tag(words)

    # We're interested in nouns (NN, NNS, NNP, NNPS represent different kinds of nouns)
    nouns = [lemmatizer.lemmatize(word.lower()) for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and word not in stop_words]

    return set(nouns)

def downsample(input_file_path, output_file_path):
    # Load the data from the input JSON file
    with open(input_file_path, "r") as f:
        data = json.load(f)

    # Create a dictionary to organize descriptions by scene_id
    scene_dict = {}
    for item in tqdm.tqdm(data, total=len(data)):
        # Check if the scene_id is already in the dictionary
        if item['scene_id'] in scene_dict:
            # If it is, append the new description to the existing list
            scene_dict[item['scene_id']].extend(nltk.sent_tokenize(item['description']))
        else:
            # If it isn't, create a new list with the description
            scene_dict[item['scene_id']] = nltk.sent_tokenize(item['description'])

    # Now create the new data for the output file
    new_data = []
    for scene_id, sentences in scene_dict.items():
        # Randomly select 30 sentences
        selected_sentences = random.sample(sentences, min(30, len(sentences)))
        # Concatenate the selected sentences into a single string
        description = ' '.join(selected_sentences)
        # Add the new data to the list
        new_data.append({"scene_id": scene_id, "description": description})

    # Finally, write the new data to the output JSON file
    with open(output_file_path, "w") as f:
        json.dump(new_data, f)


def word_centric_downsample(input_qa_file, input_caption_file, output_caption_file, description_count=15):

    def is_object_name_in_nouns(obj_name, obj_nouns):
        for noun in obj_nouns:
            if noun in obj_name:
                return True
        return False

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")

    # Read the qa json file
    with open(input_qa_file) as qa_file:
        qa_data = json.load(qa_file)

    # Read the caption json file
    with open(input_caption_file) as caption_file:
        caption_data = json.load(caption_file)

    # Create a dictionary for the caption data
    caption_dict = defaultdict(list)
    for item in caption_data:
        caption_dict[item['scene_id']].append(item)
    
    # Prepare the output data
    output_data = []

    for qa_item in tqdm.tqdm(qa_data, total=len(qa_data)):
        obj_nouns = extract_nouns(qa_item['question'], nlp)
        obj_nouns.update(extract_nouns(qa_item['situation'], nlp))
        candidate_descriptions = []
        irrelevant_descriptions = []

        for cap_item in caption_dict[qa_item['scene_id']]:
            # Normalize and remove stopwords for object name
            obj_name = " ".join([token.text for token in nlp(cap_item['object_name'].strip().lower()) if not token.is_stop])
            if is_object_name_in_nouns(obj_name, obj_nouns):
                candidate_descriptions.append(cap_item['description'])
            else:
                irrelevant_descriptions.append(cap_item['description'])

        # Randomly select 30 sentences from the candidate and irrelevant descriptions
        selected_descriptions = random.sample(candidate_descriptions, min(description_count, len(candidate_descriptions)))
        if len(selected_descriptions) < description_count:
            selected_descriptions += random.sample(irrelevant_descriptions, min(description_count - len(selected_descriptions), len(irrelevant_descriptions)))
        
        description_str = ' '.join(selected_descriptions)
        
        # Save the "question_id", "scene_id", and concatenated sentence into a json file
        output_data.append({
            'question_id': qa_item['question_id'],
            'scene_id': qa_item['scene_id'],
            'description': description_str
        })

    # Write to the output json file
    with open(output_caption_file, 'w') as output_file:
        json.dump(output_data, output_file)

def word_centric_downsample_by_sentence(input_qa_file, input_caption_file, output_caption_file, sentence_count=30):

    def is_object_name_in_nouns(obj_name, obj_nouns):
        for noun in obj_nouns:
            if noun in obj_name:
                return True
        return False

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")

    # Read the qa json file
    with open(input_qa_file) as qa_file:
        qa_data = json.load(qa_file)

    # Read the caption json file
    with open(input_caption_file) as caption_file:
        caption_data = json.load(caption_file)

    # Create a dictionary for the caption data
    caption_dict = defaultdict(list)
    for item in caption_data:
        caption_dict[item['scene_id']].append(item)
    
    # Prepare the output data
    output_data = []

    for qa_item in tqdm.tqdm(qa_data, total=len(qa_data)):
        obj_nouns = extract_nouns(qa_item['question'], nlp)
        obj_nouns.update(extract_nouns(qa_item['situation'], nlp))
        candidate_descriptions = []
        irrelevant_descriptions = []

        for cap_item in caption_dict[qa_item['scene_id']]:
            # Normalize and remove stopwords for object name
            obj_name = " ".join([token.text for token in nlp(cap_item['object_name'].strip().lower()) if not token.is_stop])
            # print(obj_name, obj_nouns)
            if is_object_name_in_nouns(obj_name, obj_nouns):
                candidate_descriptions += nltk.sent_tokenize(cap_item['description'])
                # print(nltk.sent_tokenize(cap_item['description']))
            else:
                irrelevant_descriptions+= nltk.sent_tokenize(cap_item['description'])
                # print(nltk.sent_tokenize(cap_item['description']))
        # Randomly select 30 sentences from the candidate and irrelevant descriptions
        selected_descriptions = random.sample(candidate_descriptions, min(sentence_count, len(candidate_descriptions)))
        if len(selected_descriptions) < sentence_count:
            selected_descriptions += random.sample(irrelevant_descriptions, min(sentence_count - len(selected_descriptions), len(irrelevant_descriptions)))
        
        description_str = ' '.join(selected_descriptions)
        
        # Save the "question_id", "scene_id", and concatenated sentence into a json file
        output_data.append({
            'question_id': qa_item['question_id'],
            'scene_id': qa_item['scene_id'],
            'description': description_str
        })

    # Write to the output json file
    with open(output_caption_file, 'w') as output_file:
        json.dump(output_data, output_file)

if __name__ == '__main__':
    input_qa_file = "qa/SQA_balanced_val.json"
    input_caption_file = 'scanrefer/ScanRefer_filtered_val.json'
    output_caption_file = 'scanrefer/ScanRefer_wcds30s_val.json'
    word_centric_downsample_by_sentence(input_qa_file, input_caption_file, output_caption_file)