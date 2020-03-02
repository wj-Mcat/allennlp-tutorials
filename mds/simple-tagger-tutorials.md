# 适合读者

对Pytorch熟练，想入手Allennlp框架，可对于其中的某些概念和流程都很懵*的，想实现自己的一点小`ideal`都很困难。毕竟我也是从这样的路上走过来的。

首先给大家一颗定心丸：Allennlp提供了非常多经典模型Module，比如Seq2SeqEncoder，Seq2VecEncoder，Attention等等，以及预训练模型（srl_with_elmo_luheng_2018，bert_srl_shi_2019，详细可见[Allennlp-Hub](https://github.com/allenai/allennlp-hub/blob/master/allennlp_hub/pretrained/allennlp_pretrained.py)。

如果你觉得这里的模型比较少，可结合 huggingface 旗下的 [Transformer-models](https://huggingface.co/models)，这里拥有着众多官方和大佬贡献的预训练模型，基本上可以满足你的研究和学习。不过前提是必须要对Allennlp很熟悉才行，个人感觉[transformer](https://huggingface.co/transformers/)非常简单，在`allennlp`中也是开箱即用，不存在人任何冲突。

# 概念

> 在开始正式介绍其中每个概念之前，我希望过一遍[官方Tutorial](https://allennlp.org/tutorials)和[Document-Tutorial](https://allenai.github.io/allennlp-docs/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt1/).


## Instance

标识每行数据对象，针对不同任务存储不同格式的数据。

顾名思义，每一行文本转化为Instance对象。

比如：

```javascript
instance = Instance({
    "text"    : TextField(["I", "love", "you"]),
    "label"   : LabelField("happy"),
    "tags"    : SequenceLabelField(["Person","O","Person"])
})
```
Instance对象中可针对不同任务存储不同类型的数据。

比如：文本分类，情感分类等任务，每一个`instance`都需要一个`Label`，故将其分类数据存储为`LabelField`。

比如：`POS`，`NER`，`SlotFilling` 等任务中，`Instance`中的每个`Token`都需要一个`label`，故将其分类数据存储为`SequenceLabelField`。

## DatasetReader

`dataset_reader` 负责将数据文件读取成一个`Iterable(Instance)`集合。

而我们所需要做的就是重写`_read(file_path)`函数。

因其为`Iterable`对象故可转化为List内存对象，也可作为一个 `lazy generator`，进行延迟加载数据。

## cache_path

如果`file_path`是一个网络地址，则可自动将数据下载到`cache_dir`文件夹下。然后返回本地刚下载好的数据文件路径。

否则，直接读取本地文件。

## Tokenindexer

将`Token`转化为`index`，在不同模型中，`word-level`和`character-level`是对字符进行不同程度的映射，就需要使用不同的`token_indexer`。

比如：单词`cat`在`word-level`下的index可能为`34`。可在`character-level`下的index就可能是`[23,10,18]`。

## Model

这个整个`Allennlp`框架的核心，也是我们最终模型算法的核心部分。

`Allennlp`给我们提供了多种基础模型，开箱即用，比如：`CrfTagger`，`BertForClassification`，`[BiMpm](https://arxiv.org/abs/1702.03814)`，`BidirectionalLanguageModel`，`MaskedLanguageModel` ...... 

哎呀，实在是数不过来，里面有太多提供了开箱即用的模型，希望大家能够多看[源码](https://github.com/allenai/allennlp)，了解其中最新的模型和组件。

## TextFieldEmbedder

主要是用于将`Instance`中的`TextField`转化为词向量,

**首先**，将`Instance`中的allennlp.data.fields.TextField字段转化为allennlp.data.DataArray对象。

**其次**，当我们创建TextField的时候，是有传递一个`Dict`[`str`,`allennlp.data.Tokenindexer`]对象，这样让token使用不同的方式来构建索引。比如:


```python

self.token_indexers = {
    "words": SingleIdTokenIndexer(),
    "characters":TokenCharactersIndexer()
}

instance = Instance(
    "text":TextField(["I","love","you"],self.token_indexers),
    "label":LableField("happy")
)

```

此时就会使用两种方式来对text中的tokens构建索引

最后`Dict[str,TokenEmbedder]`将其转化为词向量，注意 ⚠️ ，此处可有多个TokenEmbedders的key值与token_indexer中的key值相对应，这样不同的token经过指定的tokenindexer后生成索引后，由对应的TokenEmbedder来映射到Embedding。

最后的最后，如果有多种token_indexer/token_embedder，会自动将不同embedding拼接到一起。


*注意：token_indexers下的key必须与token_embedder下的key一致。*

这里有三种embedding，最后都会拼接成一个input-embedding，便从多种方式下获取特征。

## Modules

这个modules不同与Pytorch中的Module，而是内置来很多已经实现好的Module，提供我们在模型当中使用。

比如：多种Attention，多种Seq2SeqEncoder，多种TokenEmbedder，以及ConditionRandomField等等。对于我们复现论文模型和研究算法非常**有用**。



## Iterators

将DatasetReader读取出来的Instance转化为Batch，然后塞给Model。

Iterators也有很多类型，不同的数据组装方式对于训练的过程也是有挺大影响的。


## Trainning

对于训练的过程，在Allennlp封装的非常好，只需要一个train方法就可以完成类似于keras中的功能。

我最喜欢其中的功能就是：

- 自动生成log，这样就可以使用tensorboard查看不同的参数
- 自动保存best-checkpoint
- 生成良好的训练输出格式


# 总结

`code-tutorials`是比较适合入学者，毕竟都是从pytorch纯手动写代码过来的，如果一下子就过渡到`config & command tutorial`就会有点不适应。

不过，等对allennlp熟悉后，就会发现，`config & command`模式是真的方便，只需要配置一些参数就可以实现不同的模型，还能够生成丰富的`metrics`。
