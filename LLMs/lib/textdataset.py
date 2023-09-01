""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

import re
import os
import sys
import time
import pickle
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
#from sklearn import preprocessing
from torch.utils.data import Dataset
sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
from lib.config import CONF


class TextSQADataset(Dataset):
    def __init__(self, 
            sqa, 
            scanerefer,
            split='train', 
            tokenizer=None,
            test=False,
        ):

        # remove unanswerble qa samples for training
        self.scanerefer = {entry['scene_id']: entry['description'] for entry in scanerefer}
        self.sqa = [data for data in sqa if len(set(data['answers'])) > 0 and data['scene_id'] in self.scanerefer]

    def __len__(self):
        return len(self.sqa)

    def __getitem__(self, idx):
        scene_id = self.sqa[idx]['scene_id']       

        description = self.scanerefer[scene_id]
        situation = self.sqa[idx]['situation']
        question = self.sqa[idx]['question']
        answers = self.sqa[idx].get('answers', [])

        data_dict = {
            'scene_id': scene_id, 
            'description': description,
            'situation': situation,
            'question': question,
            'answers': answers,
        }

        return data_dict


