import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, BertTokenizer, GPT2Tokenizer, GPT2Model
import numpy as np
import random
import os
from ast import literal_eval
from tqdm import tqdm

import json
import pandas as pd
import numpy as np

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased") if os.path.isdir(os.path.join("..", "distilBERT")) else 'distilbert-base-uncased'
BERT_PATH = os.path.join("..","BERT", "bert-base-uncased") if os.path.isdir(os.path.join("..", "BERT")) else 'bert-base-uncased'
GPT2_PATH = os.path.join("..", "GPT2") if os.path.isdir(os.path.join("..", "GPT2")) else 'gpt2'

TEMPOBERT_PATH = os.path.join("..", "TempoBERT")

BERT_START_INDEX = 101
BERT_END_INDEX = 102

class NytDataset(Dataset):

    def __init__(self, data_dir, encoder, time_precision, train, val=False, axis='authors', max_len = 512, seed = 1):
        super(NytDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.val = val
        self.seed = seed
        self.max_len = max_len
        self.encoder = encoder
        self.axis = axis
        self.time_precision = time_precision

        with open(self.data_dir, 'r') as f:
            corpus = f.readlines()

        self.data = []
        for c in corpus:
            self.data.append(json.loads(c))

        self.data = pd.DataFrame(self.data)
        self.data = self.data.explode('authors').explode('texts')

        self.data = self.data.explode(self.axis)

        self.n_axis = len(self.data[self.axis].unique())

        self.data['date'] = pd.to_datetime(self.data.date, format='%Y-%m-%d')

        self.min_date = min(self.data.date)
        self.max_date = max(self.data.date)

        if encoder == "DistilBERT":
            self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        elif "TempoBERT" in encoder:
            self.tokenizer = BertTokenizer.from_pretrained(encoder)
        elif encoder == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH)

        if self.train:

            tokens = sorted(['<' + s + '>' for s in self.data.date.dt.strftime('%Y').unique()])
            num_tokens = self.tokenizer.add_tokens(tokens)
            print(f"Added {num_tokens} time tokens to the vocabulary !")

            with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'train.txt'), 'r') as f:
                train_ids = [i.strip() for i in f.readlines()]

            self.data = self.data[self.data.id.isin(train_ids)]

            self.data["texts"] = "<" + self.data.date.dt.year.astype(str) + "> " + self.data.texts
            
        else:
            if self.val:
                with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'val.txt'), 'r') as f:
                    test_ids = [i.strip() for i in f.readlines()]
            else:
                with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'test.txt'), 'r') as f:
                    test_ids = [i.strip() for i in f.readlines()]

            self.data = self.data[self.data.id.isin(test_ids)]

        self.data = self.data.sort_values([self.axis, 'date'])

        self.axis2id = {a:i for i, a in enumerate(self.data[self.axis].unique())}
        self.id2axis = {i:a for a,i in self.axis2id.items()}

        self.data['ddelta'] = (self.data.date - self.min_date).dt.days

        self.data['timestep'] = (self.data.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

        self.data['start_pin'] = self.data.groupby(self.axis)['ddelta'].transform('min')
        self.data['end_pin'] = self.data.groupby(self.axis)['ddelta'].transform('max')

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")
        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)
        self.data['label'] = 1

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'timestep', 'ddelta', 'start_pin', 'end_pin', 'label']].to_dict('records')

        print("There are %d texts processed in corpus (train = %d)" % (len(self.processed_data), train))

    def interpolate_axis(self, dynamic, n_steps=100):

        self.interpolation_df = self.data[[self.axis, 'axis_id', 'date']].drop_duplicates(self.axis, keep='last')
        self.interpolation_df = self.interpolation_df.reset_index(drop=True)

        if dynamic=="local":
            self.interpolation_df['mindate'] = list(self.data.drop_duplicates(self.axis)['date'])
            self.interpolation_df['maxdate'] = self.interpolation_df['date']
            self.interpolation_df['ddelta'] = (self.interpolation_df.maxdate - self.interpolation_df.mindate).dt.days

            self.interpolation_df = self.interpolation_df.loc[self.interpolation_df.index.repeat(n_steps+1)]
            self.interpolation_df['day_factor'] = [i/(n_steps+1) for i in range(n_steps+1)]*self.n_axis

            self.interpolation_df['date'] = self.interpolation_df.mindate + pd.to_timedelta(self.interpolation_df['day_factor'] * self.interpolation_df['ddelta'], unit='d')
        elif dynamic=="global":
            
            self.interpolation_df = self.interpolation_df.loc[self.interpolation_df.index.repeat(n_steps+1)]

            self.interpolation_df['date'] = [i/(n_steps+1)*self.data.ddelta.max() for i in range(n_steps+1)]*self.n_axis
            self.interpolation_df['date'] = pd.to_timedelta(self.interpolation_df['date'], unit='d') + self.min_date

        self.interpolation_df['timestep'] = (self.interpolation_df.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, truncation=True, max_length = self.max_len, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        item = self.processed_data[index]
        doc_num = item['doc_id']

        if doc_num == 0:
            index+=2
        if doc_num == 1:
            index+=1

        item = self.processed_data[index]
        doc_num = item['doc_id']

        T = doc_num

        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]
        y_t = self.processed_data[index - T + t2]
        y_T = self.processed_data[index]

        t_ = t1
        t = t2

        total_docs = item['corpus_length']
        result = {
            'y_0': y_0['texts'],
            'y_t': y_t['texts'],
            'y_T': y_T['texts'],
            't_': y_0['ddelta'],
            't': y_t['ddelta'],
            'T': y_T['ddelta'],
            'total_t': total_docs,
            'class_t_': y_0['timestep'],
            'class_t': y_t['timestep'],
            'class_T': y_T['timestep'],
            'axis':y_0['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)


class S2gDataset(Dataset):

    def __init__(self, data_dir, encoder, time_precision, train, val=False, axis='authors', max_len = 512, seed = 1):
        super(S2gDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.val = val
        self.seed = seed
        self.max_len = max_len
        self.encoder = encoder
        self.axis = axis
        self.time_precision = time_precision

        with open(self.data_dir, 'r') as f:
            corpus = f.readlines()

        self.data = []
        for c in corpus:
            self.data.append(json.loads(c))

        self.data = pd.DataFrame(self.data)
        self.data = self.data.explode('authors').explode('texts')

        self.data = self.data.explode(self.axis)

        self.n_axis = len(self.data[self.axis].unique())

        self.data['date'] = pd.to_datetime(self.data.date, format='%Y')

        self.data = self.data.sort_values([axis, 'date'])

        self.min_date = min(self.data.date)
        self.max_date = max(self.data.date)

        if encoder == "DistilBERT":
            self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        elif "TempoBERT" in encoder:
            self.tokenizer = BertTokenizer.from_pretrained(encoder)
        elif encoder == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH)

        self.axis2id = {a:i for i, a in enumerate(self.data[self.axis].unique())}
        self.id2axis = {i:a for a,i in self.axis2id.items()}

        if self.train:

            tokens = sorted(['<' + s + '>' for s in self.data.date.dt.strftime('%Y').unique()])
            num_tokens = self.tokenizer.add_tokens(tokens)
            print(f"Added {num_tokens} time tokens to the vocabulary !")

            with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'train.txt'), 'r') as f:
                train_ids = [i.strip() for i in f.readlines()]

            self.data = self.data[self.data.id.isin(train_ids)]
            
            self.data["texts"] = "<" + self.data.date.dt.year.astype(str) + "> " + self.data.texts

        else:
            if self.val:
                with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'val.txt'), 'r') as f:
                    test_ids = [i.strip() for i in f.readlines()]
            else:
                with open(os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), 'test.txt'), 'r') as f:
                    test_ids = [i.strip() for i in f.readlines()]

            self.data = self.data[self.data.id.isin(test_ids)]

        self.data = self.data.sort_values([axis, 'date'])

        self.data['ddelta'] = (self.data.date - self.min_date).dt.days

        self.data['timestep'] = (self.data.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

        self.data['start_pin'] = self.data.groupby(self.axis)['ddelta'].transform('min')
        self.data['end_pin'] = self.data.groupby(self.axis)['ddelta'].transform('max')

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")
        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)
        self.data['label'] = 1

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'timestep', 'ddelta', 'start_pin', 'end_pin', 'label']].to_dict('records')

        print("There are %d texts processed in corpus (train = %d)" % (len(self.processed_data), train))

    def interpolate_axis(self, dynamic, n_steps=100):

        self.interpolation_df = self.data[[self.axis, 'axis_id', 'date']].drop_duplicates(self.axis, keep='last')
        self.interpolation_df = self.interpolation_df.reset_index(drop=True)

        if dynamic=="local":
            self.interpolation_df['mindate'] = list(self.data.drop_duplicates(self.axis)['date'])
            self.interpolation_df['maxdate'] = self.interpolation_df['date']
            self.interpolation_df['ddelta'] = (self.interpolation_df.maxdate - self.interpolation_df.mindate).dt.days

            self.interpolation_df = self.interpolation_df.loc[self.interpolation_df.index.repeat(n_steps+1)]
            self.interpolation_df['day_factor'] = [i/(n_steps+1) for i in range(n_steps+1)]*self.n_axis

            self.interpolation_df['date'] = self.interpolation_df.mindate + pd.to_timedelta(self.interpolation_df['day_factor'] * self.interpolation_df['ddelta'], unit='d')
        elif dynamic=="global":
            
            self.interpolation_df = self.interpolation_df.loc[self.interpolation_df.index.repeat(n_steps+1)]

            self.interpolation_df['date'] = [i/(n_steps+1)*self.data.ddelta.max() for i in range(n_steps+1)]*self.n_axis
            self.interpolation_df['date'] = pd.to_timedelta(self.interpolation_df['date'], unit='d') + self.min_date

        self.interpolation_df['timestep'] = (self.interpolation_df.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, truncation=True, max_length = self.max_len, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        item = self.processed_data[index]
        doc_num = item['doc_id']

        if doc_num == 0:
            index+=2
        if doc_num == 1:
            index+=1

        item = self.processed_data[index]
        doc_num = item['doc_id']

        T = doc_num

        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]
        y_t = self.processed_data[index - T + t2]
        y_T = self.processed_data[index]

        t_ = t1
        t = t2

        total_docs = item['corpus_length']
        result = {
            'y_0': y_0['texts'],
            'y_t': y_t['texts'],
            'y_T': y_T['texts'],
            't_': y_0['ddelta'],
            't': y_t['ddelta'],
            'T': y_T['ddelta'],
            'total_t': total_docs,
            'class_t_': y_0['timestep'],
            'class_t': y_t['timestep'],
            'class_T': y_T['timestep'],
            'axis':y_0['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)

class VarNytDataset(NytDataset):

    def sample_negative(self, n_neg=5):

        print("Negative sampling ...")
        neg_examples = []
        for example in tqdm(self.processed_data):

            axis_id = example["axis_id"]
            timestep = example["timestep"]

            neg_axis = self.data[(self.data.axis_id != axis_id)|((self.data.axis_id == axis_id) & (self.data.timestep!=timestep))].sample(n_neg)

            # These are negative pairs, so the texts is the "good" example
            # but either axis or timestep is bad
            neg_axis["texts"] = example["texts"]

            neg_axis["label"] = 0

            neg_examples.extend(neg_axis[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'timestep', 'ddelta', 'start_pin', 'end_pin', 'label']].to_dict('records'))

        self.processed_data.extend(neg_examples)

        print("There are %d training pairs after negative sampling." % (len(self.processed_data)))

    def __getitem__(self, index):
        item = self.processed_data[index]

        total_docs = item['corpus_length']
        result = {
            'y_t': item["texts"],
            'label': item["label"],
            't': item['ddelta'],
            'total_t': total_docs,
            'timestep': item['timestep'],
            'axis':item['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)

class VarS2gDataset(S2gDataset):

    def sample_negative(self, n_neg=5):

        print("Negative sampling ...")
        neg_examples = []
        for example in tqdm(self.processed_data):

            axis_id = example["axis_id"]
            timestep = example["timestep"]

            neg_axis = self.data[(self.data.axis_id != axis_id)|((self.data.axis_id == axis_id) & (self.data.timestep!=timestep))].sample(n_neg)

            # These are negative pairs, so the texts is the "good" example
            # but either axis or timestep is bad
            neg_axis["texts"] = example["texts"]

            neg_axis["label"] = 0

            neg_examples.extend(neg_axis[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'timestep', 'ddelta', 'start_pin', 'end_pin', 'label']].to_dict('records'))

        self.processed_data.extend(neg_examples)

        print("There are %d training pairs after negative sampling." % (len(self.processed_data)))

    def __getitem__(self, index):
        item = self.processed_data[index]

        total_docs = item['corpus_length']
        result = {
            'y_t': item["texts"],
            'label': item["label"],
            't': item['ddelta'],
            'total_t': total_docs,
            'timestep': item['timestep'],
            'axis':item['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)