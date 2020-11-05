#! -*- coding: utf-8 -*-
# 用tokenizer对语料分词，然后统计每个token的词频

import glob, json, re
import sentencepiece as spm
from bert4keras.snippets import parallel_apply
from tqdm import tqdm


spm_path = '/root/kg/bert/mt5/sentencepiece.model'
sp_model = spm.SentencePieceProcessor()
sp_model.Load(spm_path)
global_tokens = {}


def corpus():
    filenames = glob.glob('/root/data_pretrain/*/*')
    count, texts = 0, []
    for filename in filenames:
        with open(filename) as f:
            for l in f:
                l = json.loads(l)['text'].strip()
                texts.append(l)
                count += 1
                if count == 1000:
                    yield texts
                    count, texts = 0, []
    if texts:
        yield texts


def count(texts):
    tokens = {}
    for text in texts:
        for t in sp_model.encode_as_pieces(text):
            tokens[t] = tokens.get(t, 0) + 1
    return tokens


def callback(tokens):
    for k, v in tokens.items():
        global_tokens[k] = global_tokens.get(k, 0) + v


parallel_apply(
    func=count,
    iterable=tqdm(corpus()),
    workers=20,
    max_queue_size=1000,
    callback=callback,
)


import pandas as pd

dic = pd.Series(global_tokens).sort_values(ascending=False)
dic.to_csv('result.csv', header=None, encoding='utf-8', sep='\t')
json.dump(global_tokens, open('result.json', 'w'))
