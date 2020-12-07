#! -*- coding: utf-8 -*-
# 根据count的结果精简sentencepiece模型
# 注：需要 sentencepiece>=0.1.94

from tqdm import tqdm
import json
import pandas as pd
from sentencepiece import sentencepiece_model_pb2 as model
import sentencepiece as spm

min_count = 1000
old_model = '/root/kg/bert/mt5/sentencepiece.model'
new_model = '/root/kg/bert/mt5/sentencepiece_cn.model'
new_model_keep_tokens = '/root/kg/bert/mt5/sentencepiece_cn_keep_tokens.json'

dic = json.load(open('result.json'))
dic = pd.Series(dic).sort_values(ascending=False)
dic = dic[dic >= min_count]
dic = set(dic.index)

m = model.ModelProto()
m.ParseFromString(open(old_model, 'rb').read())
pieces = m.pieces[:259] + [p for p in m.pieces[259:] if p.piece in dic] + m.pieces[-100:]

for i in tqdm(range(len(m.pieces))):
    del m.pieces[-1]

m.pieces.extend(pieces)

with open(new_model, 'wb') as f:
    f.write(m.SerializeToString())

sp1 = spm.SentencePieceProcessor()
sp2 = spm.SentencePieceProcessor()

sp1.load(old_model)
sp2.load(new_model)

keep_tokens = []

for i in range(sp2.get_piece_size()):
    keep_tokens.append(sp1.piece_to_id(sp2.id_to_piece(i)))

keep_tokens.append(sp1.get_piece_size())
keep_tokens.append(sp1.get_piece_size() + 1)

with open(new_model_keep_tokens, 'w') as f:
    json.dump(keep_tokens, f)
