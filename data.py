import pandas as pd

import torchtext
from torchtext.vocab import GloVe

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from transformers import DistilBertTokenizer

import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw


def retrieve_data():
    data = pd.read_csv('ielts_writing_dataset.csv')
    data = data[['Task_Type', 'Question', 'Essay', 'Overall']]
    return data

def tokenize_data(data: pd.DataFrame, max_e_length = 300, max_q_length = 128):
    q = data['Question'].tolist()
    e = data['Essay'].tolist()
    
    essay_temp = []
    question_temp = []
    max_e = 0
    max_q = 0


    tokenized_data = []
    d = {}

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    for essay_content, question_content in zip(e, q):
        text = tokenizer(essay_content, padding = 'max_length', truncation = True, max_length = 300,return_tensors = 'pt') #TL 
        essay_temp.append(text['input_ids'][0])              #TL

    data['E_tokenized'] = essay_temp


    return data
                               
def get_data_loader(data, batch_size: int = 128):
    data = data[(data['Overall'] != 1.0) & (data.Overall != 3.0) & (data.Overall != 3.5)]

    #-------------------------------FOR CLASSIFICATION-----------------------
    data['Overall'] = data['Overall'].replace({4.0 : '<6.5', 4.5: '<6.5', 5.0 : '<6.5', 5.5: '<6.5', 6.0: '<6.5', 6.5: '>=6.5', 7.0 : '>=6.5', 7.5 : '>=6.5', 8.0: '>=6.5', 8.5 : '>=6.5', 9.0 : '>=6.5'})
    #----------------------------------------------------------------------

    d = {}
    for i in range(len(data.Overall)):
        if data.Overall.iloc[i] not in d:
            d[data.Overall.iloc[i]] = [i]
        else:
            d[data.Overall.iloc[i]].append(i)

    train_indices = []
    val_indices = []
    test_indices = []

    split1 = .75
    split2 = .95

    for k,v in d.items():
        train_indices += v[:int(split1*len(v))]
        val_indices += v[int(split1*len(v)) : int(split2*len(v))]
        test_indices += v[int(split2*len(v)):]


    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    #-----------FOR CLASSIFICATION-----------------
    data['Overall'] = data['Overall'].apply(lambda x: str(x))
    target_classes = {'<6.5': 0, '>=6.5' : 1}
    data['Overall'] = data['Overall'].replace(target_classes)
    #-----------------------------------------------

    data = data[['E_tokenized', 'Overall']].values.tolist()

    train_loader = DataLoader(data, batch_size= batch_size, sampler = train_sampler)
    val_loader = DataLoader(data, batch_size = batch_size, sampler = val_sampler)
    test_loader = DataLoader(data, batch_size = batch_size, sampler= test_sampler)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data = retrieve_data()
    tokenized_data, embed = tokenize_data(data)
    train_loader, val_loader, test_loader = get_data_loader(tokenized_data)
