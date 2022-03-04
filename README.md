# T5 in bert4keras
整理一下在keras中使用T5模型的要点，尤其是中文场景下的使用要点。以多国语言版mT5为例。

博客链接：https://kexue.fm/archives/7867

本项目实验环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.1

## 模型下载

首先，要想办法下载Google开放的权重，最简单的方式，是找一台能科学上网的服务器，在上面安装gsutil，然后执行
```shell
gsutil cp -r gs://t5-data/pretrained_models/mt5/small .
```

T5使用sentencepiece作为tokenizer，mT5的tokenizer模型下载地址为
```shell
gsutil cp -r gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model .
```

笔者精简好的tokenizer文件：[sentencepiece_cn.model](https://github.com/bojone/t5_in_bert4keras/blob/main/tokenizer/sentencepiece_cn.model)和[sentencepiece_cn_keep_tokens.json](https://github.com/bojone/t5_in_bert4keras/blob/main/tokenizer/sentencepiece_cn_keep_tokens.json)

另外，为了方便国内用户，笔者将small版和base版整理分享到[百度网盘](https://pan.baidu.com/s/1YWaStqB6Epkxfyx6WcOzWw)(mwfc)了。

## Config

T5模型的配置文件是gin格式的，这不符合bert4keras的输入，使用者请根据所给的gin和下述模版构建对应的config.json文件。

下面是mT5 small版的参考config.json：
```python
{
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1024,
  "num_attention_heads": 6,
  "attention_head_size": 64,
  "num_hidden_layers": 8,
  "vocab_size": 250112,
  "hidden_act": ["gelu", "linear"]
}
```

一般要修改的是`hidden_size`、`intermediate_size`、`num_attention_heads`、`attention_head_size`和`num_hidden_layers`这几个参数。

## 基本使用

```python
# 模型路径
config_path = '/root/kg/bert/mt5/mt5_small/t5_config.json'
checkpoint_path = '/root/kg/bert/mt5/mt5_small/model.ckpt-1000000'
spm_path = '/root/kg/bert/mt5/sentencepiece.model'

# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='&lt;/s&gt;')

# 加载模型
t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model
```

## 中文优化

```python
# 模型路径
config_path = '/root/kg/bert/mt5/mt5_base/t5_config.json'
checkpoint_path = '/root/kg/bert/mt5/mt5_base/model.ckpt-1000000'
spm_path = '/root/kg/bert/mt5/sentencepiece_cn.model'
keep_tokens_path = '/root/kg/bert/mt5/sentencepiece_cn_keep_tokens.json'

# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='&lt;/s&gt;')
keep_tokens = json.load(open(keep_tokens_path))

# 加载模型
t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model
```

细节请参考：[task_autotitle_csl.py](https://github.com/bojone/t5_in_bert4keras/blob/main/task_autotitle_csl.py)。

## 交流联系
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
