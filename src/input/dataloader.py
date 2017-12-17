import os
import sys

from input.vocab import Vocab
from input.glove import Glove
from input.dataset import Dataset
from input.config import *

#from vocab import Vocab
#from glove import Glove
#from dataset import Dataset
#from config import *

import torch
import torch.utils.data as data

#train loader
def get_train_data(files, glove_file, batch_size=1):
    vocab = Vocab(files)
    vocab.add_padunk_vocab()
    vocab.create()

    glove = Glove(glove_file)
    glove.create(vocab)

    dataset = Dataset(files)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dataloader, vocab, glove

#test loader
def get_test_data(files, vocab, glove, batch_size=1):
    dataset = Dataset(files)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dataloader


if __name__=='__main__':
    file_names = ['train.json']
    cur_dir = os.getcwd()
    files = [os.path.join(cur_dir,file) for file in file_names]
    glove_dir = './glove'
    glove_filename = 'glove.6B.50d.txt'
    glove_filepath = os.path.join(glove_dir, glove_filename)

    train_dataloader, vocab, glove = get_train_data(files, glove_filepath, 2)
    '''for batch in train_dataloader:
        print (batch[2])
        print (len(batch[2][0]))
        print (len(batch[2][1]))
        print (batch[2][0][0])
        print (batch[2][1][0])
        break'''
    file_names = ['dev.json']
    cur_dir = os.getcwd()
    files = [os.path.join(cur_dir,file) for file in file_names]
    dev_dataloader = get_test_data(files, vocab, glove, 2)
    for batch in dev_dataloader:
        print (batch[4])
        print (len(batch[4][0]))
        print (len(batch[4][1]))
        print (batch[4][0][4])
        print (batch[4][1][4])
        print (batch[6][0][5])
        print (batch[6][1][5])
        print (len(batch[0]))
        print (len(batch[0][0]))
        print (len(batch[0][0][0]))
        break
