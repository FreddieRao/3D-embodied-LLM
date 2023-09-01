import os
import re  # we will use regular expressions to extract the time from the error message
import sys
import json
import argparse
import collections
import torch
import numpy as np
import random
import time
import tqdm
import openai
import string
openai.organization = "org-VfrZJM0sEx0WGJG8feSW0vJ6"
openai.api_key = "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"
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from lib.textdataset import TextSQADataset
from collections import OrderedDict

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument("--llm_type", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--ckpt", type=str, help="checkpoint to evaluate")
    args = parser.parse_args()
    return args

def get_dataloader(args, SQA_TRAIN, SQA_VAL, SCANREFER_TRAIN, SCANREFER_VAL, split='val', test=True):

    tokenizer = None

    dataset = TextSQADataset(
        sqa=SQA_VAL if test else SQA_TRAIN,
        scanerefer=SCANREFER_VAL if test else SCANREFER_TRAIN,
        split=split,
        tokenizer=tokenizer,
        test=test
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return dataloader

def gpt_api_forward(content, args, test=True):
    # record the time before the request is sent
    start_time = time.time()

    # send a ChatCompletion request to count to 100
    response = openai.ChatCompletion.create(
        model=args.llm_type,
        messages=[
            {'role': 'user', 'content': content}
        ],
        temperature=0,
        top_p = 1,
        n = 1,
    )

    reply_content = response['choices'][0]['message']['content']
    # calculate the time it took to receive the response
    response_time = time.time() - start_time

    return response_time, reply_content

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, SCANREFER_TRAIN, SCANREFER_VAL, path, answer_counter_list):

    # Set up Dataloader and Inference Model
    val_dataloader = get_dataloader(args, SQA_TRAIN, SQA_VAL, SCANREFER_TRAIN, SCANREFER_VAL, split='val', test=True)
    model_f = gpt_api_forward

    # Context: There is a book on the desk. A laptop with a green cover is to the left of the book.\n Q: I\’m working by the desk. What is on the desk beside the book? \n A: laptop\n
    # Context:{ }\n Q: { } { } \n A: 

    # Evaluation Metrics
    # read the first (and only) line, and split it into parts
    with open("dump/eval_buffer_0.txt", "r") as file:
        parts = file.readline().split()
        right_count = int(parts[0])
        count = int(parts[1])

    pbar = tqdm.tqdm(val_dataloader, desc="Evaluating")

    for i, data in enumerate(pbar):
        if i < count:
            continue
        while True:
            try:
                # test_context = "Please answer the following question in one word. Here is an example. \n Context: There is a book on the desk. A laptop with a green cover is to the left of the book.\n Q: I\'m working by the desk. What is on the desk beside the book? \n A: laptop\n" +\
                #         "Context:{}\n Q: {} {} \n A:".format(data['description'], data['situation'], data['question'])
                test_context = "Please answer the following question using one word. \n CONTEXT: There is a book on the desk. A laptop with a green cover is to the left of the book.\n Q: I\'m working by the desk. What is on the desk beside the book? \n A: laptop\n" +\
                        "CONTEXT:{}\n Q: {} {} \n A:".format(data['description'], data['situation'], data['question'])
                gt_answer = data['answers'][0][0] #TODO  List of questions
                test_time, pred_answer = model_f(test_context, args) # ERROR CASTING

                if normalize_answer(pred_answer) == (gt_answer):
                    right_count += 1
                count += 1
                pbar.set_description(f'Processing data (Accuracy: {right_count / count})')
                break

            except Exception as e:
                with open("dump/eval_buffer_0.txt", "w") as file:
                    file.write(str(right_count) + " " + str(count) + "\n")
                print("Exception occurred:", e)
                print("Waiting for 4 seconds before retrying...")
                time.sleep(4)  # Wait for 5 seconds
    pbar.close()

    return right_count / count

if __name__ == "__main__":
    args = parse_option()
    # torch.cuda.set_device('cuda:{}'.format(args.gpu))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # setting dataaset
    project_name = "SQA_balanced"
    SQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_train.json")))
    SQA_VAL = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_val.json")))
    SQA_TEST = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_test.json")))
    answer_counter_list = json.load(open(os.path.join(CONF.PATH.SQA, "answer_counter.json")))

    # SCANREFER_TRAIN = json.load(open(os.path.join(CONF.SCANREFER_TRAIN)))
    SCANREFER_TRAIN = None
    SCANREFER_VAL = json.load(open(os.path.join(CONF.SCANREFER_VAL)))

    # reproduce
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # test
    path = args.ckpt
    save_list = test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, SCANREFER_TRAIN, SCANREFER_VAL, path, answer_counter_list)