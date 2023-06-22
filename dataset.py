import os
import cv2
import torch
import numpy as np
from os.path import join
from collections import defaultdict
from torch.utils.data import Dataset


class HandwritingDataset:
    def __init__(self, root="datasets"):
        member = os.listdir(root)
        path = defaultdict(defaultdict)

        for m in member:
            for char in os.listdir(join(root, m)):
                ls = sorted(os.listdir(join(root, m, char)))
                pls = [x for i, x in enumerate(ls) if i % 9 == 0]
                path[m][char] = list(map(lambda x: join(root, m, char, x), pls))

        print(f"Who are the members? {list(path.keys())}\n")

        for m in member:
            print(
                f"Member {m:5s} contributed {sum([len(x) for x in path[m].values()])} characters including classes: {list(path[m])}"
            )

        self.path = path
        self.id2cls = {}
        self.x = []
        self.y = []
        self.char = []
        for i, m in enumerate(path.keys()):
            self.id2cls[i] = m
            for char in path[m].keys():
                for c in path[m][char]:
                    self.x.append(
                        np.expand_dims(cv2.imread(c, 0), -1).transpose(2, 0, 1)
                    )
                    self.y.append(i)
                    self.char.append(char)

        self.x = torch.from_numpy(np.array(self.x))
        self.y = torch.tensor(self.y)
        self.char = np.array(self.char)

        print(
            f"Constructed members' handwriting dataset for {len(path.keys())} members with total {len(self.char)} characters"
        )

    def get(self, cls: int, character: str):
        """cls: 0 (yeow), 1 (chua) ..., character: 0, 1, 2, 3 ...
        return: a list of index self.__getitem__
        """
        cls_index = torch.where(self.y == cls)[0].numpy()
        char_index = np.where(self.char == character)[0]
        idx = cls_index[np.in1d(cls_index, char_index)]
        return idx

    def stratified_split(self, train_size=0.95):
        cls = self.get_cls()
        chars = self.get_chars()

        m = int(len(chars) * train_size)

        train = []
        test = []
        for i, (cls_idx, _cls) in enumerate(cls.items()):
            for j, char in enumerate(chars):
                train_index = self.get(cls_idx, char)[:m]
                test_index = self.get(cls_idx, char)[m:-1]
                train.extend(train_index)
                test.extend(test_index)

        return (self.x[train], self.y[train]), (self.x[test], self.y[test])
        # return (self.char[train], (self.x[train], self.y[train])), (self.char[test], (self.x[test], self.y[test]))

    def get_chars(self):
        return np.unique(self.char).tolist()

    def get_cls(self):
        return self.id2cls

    def get_class_label(self, cls):
        return self.id2cls[cls.item()]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.char[idx], (self.x[idx], self.y[idx])


class PrepForDataLoader(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        x, y = self.dataset
        return x[idx].float(), y[idx]
        # char, (x, y) = self.dataset
        # return char[idx], (x[idx], y[idx])
