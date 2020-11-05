# T5 in bert4keras
整理一下在keras中使用T5模型的要点，尤其是中文场景下的使用要点。以多国语言版mT5为例。

博客链接：https://kexue.fm/archives/7867

## 权重下载

首先，要想办法下载Google开放的权重，最简单的方式，是找一台能科学上网的服务器，在上面安装gsutil，然后执行
```shell
gsutil cp -r gs://t5-data/pretrained_models/mt5/small .
```

## config

T5模型的配置文件是gin格式的，这不符合bert4keras的输入，使用者请根据下述模版构建对应的config.json文件。

下面是mT5 small版的参考config.json：
```python
{
  "hidden_act": "gelu",
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

## 中文优化

## 交流联系
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
