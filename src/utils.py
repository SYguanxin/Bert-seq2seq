import json
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

from transformers import BertTokenizerFast


class Noval_Dataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 tokenizer=None,
                 data_collator=None,
                 long_target=True,
                 ):
        super(Noval_Dataset, self).__init__()
        self.train = train
        self.transform = transform
        if tokenizer is None:
            tokenizer = BertTokenizerFast.from_pretrained('src/bert-base-chinese')
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.eos_token = tokenizer.sep_token
        self.tokenizer = tokenizer
        self.DataCollator = data_collator
        self.input_par = 3
        if long_target:
            self.label_par = 8
            self.max_target_len = 512
        else:
            self.label_par = 1
            self.max_target_len = 256

        file_path = root + f'/large_train_data.json'
        if not train:
            file_path = root + '/test.json'
        with open(file_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        drop_char = '[\u3000\xa0 ….—]+'
        novals = [[[re.sub('[“”]', '"', re.sub(drop_char, '', seq)) for seq in item['content']]
                   for item in book] for book in data.values()][:50]
        titles = [[re.sub(drop_char, '', item['title']) for item in book] for book in data.values()][:50]

        step = min(self.input_par, self.label_par)
        self.inputs = [title + '[SEP]' + ''.join(chapter[idx:idx+self.input_par])
                       for noval, title in zip(novals, titles)
                       for chapter, title in zip(noval, title) if len(chapter) >= self.label_par + self.input_par
                       for idx in range(0, len(chapter) - self.label_par - self.input_par + 1, step)]
        target = [''.join(chapter[idx:idx+self.label_par]) + '[SEP]' for noval in novals
                  for chapter in noval if len(chapter) >= self.label_par + self.input_par
                  for idx in range(self.input_par, len(chapter) - self.label_par + 1, step)]

        self.inputs = self.tokenizer(self.inputs, max_length=512, truncation=True)
        target = self.tokenizer(target, max_length=self.max_target_len, truncation=True, add_special_tokens=False)
        self.inputs['labels'] = target['input_ids']
        self.inputs['num_tokens'] = [len(seq) for seq in target['input_ids']]

    def __len__(self):
        return len(self.inputs['labels'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.inputs.items()}

    def collate(self, batch):
        nt = torch.LongTensor([item['num_tokens'] for item in batch])
        data = []
        for item in batch:
            item.pop('num_tokens')
            data.append(item)
        data = self.DataCollator(data)
        data['num_tokens'] = nt
        return data


if __name__ == '__main__':
    Noval_Dataset(root='data')