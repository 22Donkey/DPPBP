import pickle
import torch
import torch.utils.data as Data
import numpy as np

from configuration import config
from util import util_file


def toInt(l):
    l = [int(i) for i in l]
    return l

def make_data_with_unified_length(seq_list, labels, config):
    data = []
    for i in range(len(labels)):
        labels[i] = [0] + labels[i] + [0]
        # pssm[i] = pssm_pad + pssm[i] +pssm_pad
        data.append([seq_list[i], labels[i]])
    return data



def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size


    input_ids, labels, pssm = zip(*data)

    if cuda:
        input_ids, labels, pssm = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels), torch.cuda.LongTensor(pssm)
    else:
        input_ids, labels, pssm = torch.LongTensor(input_ids), torch.LongTensor(labels), torch.LongTensor(pssm)

    print('-' * 20, '[construct_dataset]: check data device', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_sequences, seq_len]
    print('labels:', labels.shape)  # [num_sequences, seq_len]

    data_loader = Data.DataLoader(MyDataSet(input_ids, labels, pssm),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    print('len(data_loader)', len(data_loader))
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, labels, pssm):
        self.input_ids = input_ids
        self.labels = labels
        self.pssm = pssm

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.pssm[idx]


def load_data(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    sequences_train, labels_train = util_file.load_tsv_format_data(path_data_train)
    sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)

    data_train = make_data_with_unified_length(sequences_train, labels_train, config)
    data_test = make_data_with_unified_length(sequences_test, labels_test, config)
    return data_train, data_test

