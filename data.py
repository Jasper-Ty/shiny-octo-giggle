import os
from io import open

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

tok2idx = {str(i): i for i in range(0, 7)}

class Data:
    def __init__(self, path):
        self.test_src, self.test_tgt = self.parse(os.path.join(path, 'test.txt'))
        self.train_src, self.train_tgt = self.parse(os.path.join(path, 'train.txt'))
        self.validate_src, self.validate_tgt = self.parse(os.path.join(path, 'validate.txt'))

        self.ntest = len(self.test_src)
        self.ntrain = len(self.train_src)
        self.nvalidate = len(self.validate_src)

    def line2tensor(self, line):
        return torch.tensor([tok2idx[tok] for tok in line.split()]).reshape(1, -1)

    def parse(self, path):
        assert os.path.exists(path)

        src = []
        tgt = []

        with open(path, "r", encoding="utf8") as f:
            it = iter (f)

            for _, line in enumerate(it):
                cycle = self.line2tensor(line)
                permutation = self.line2tensor(next(it))

                src.append(cycle)
                tgt.append(permutation)

        src = torch.cat(src, 0)
        tgt = torch.cat(tgt, 0)

        return (src, tgt)

    def test_dataloader(self, batch_size):
        test_data = TensorDataset(torch.LongTensor(self.test_src), torch.LongTensor(self.test_tgt))

        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        return test_dataloader

    def train_dataloader(self, batch_size):
        train_data = TensorDataset(torch.LongTensor(self.train_src), torch.LongTensor(self.train_tgt))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        return train_dataloader

    def validate_dataloader(self, batch_size):
        validate_data = TensorDataset(torch.LongTensor(self.validate_src), torch.LongTensor(self.validate_tgt))

        validate_sampler = RandomSampler(validate_data)
        validate_dataloader = DataLoader(validate_data, sampler=validate_sampler, batch_size=batch_size)

        return validate_dataloader

