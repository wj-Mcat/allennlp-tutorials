from typing import Iterator, List, Dict
# AllenNlp 基于Pytorch编写，所以几乎可以在Allennlp中使用pytorch所有组件
# eg：modules，optimizer，operation ......
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from overrides import overrides
torch.manual_seed(1)


class PosDatasetReader(DatasetReader):
    """
    读取数据文件，格式如下：
    
    The###DET dog###NN ate###V the###DET apple###NN
    
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        # token_indexs 将token映射到指定的索引上
        # 如果未指定token_indexers，则默认将每个单词映射到唯一id
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    @overrides
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        
        # 只有TextField上才会传递token_indexers
        # LabelField 和 SequenceLabelField 不传递 token_indexers
        sentence_field = TextField(tokens, self.token_indexers)
        
        # fields 最后是转化为Instance
        # 同时将文本数据（TextField）放置在 "sentence" 键上，所以在模型的forward函数上，就应该有sentence参数。
        
        fields = {"sentence": sentence_field}
        # tags这里为什么可能是None？
        # 答：在train模式下就会传递tags数据，可如果是在对应的predict模型下，就不会喘息tags。
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)



class LstmTagger(Model):
    """
    最上层的模型，融合多种modules
    """
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        
        # 用于将token_index转化为embedding
        self.word_embeddings = word_embeddings
        
        # 派生于Seq2SeqEncoder，建议看看allennlp.modules.seq2seq下的多种模型
        # 官网上的api文档写的太简单了，看源码你会了解的更多。
        self.encoder = encoder
        
        # allennlp有一个很让人舒适的地方就是：大部分allennlp.modules下的module，都有一个`get_output_dim()`函数，
        # 这样在一定程度上减少模型的耦合度和配置复杂性
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        
        # 不是Loss，也不是Optimizer，不影响训练的learning-rate或grad
        # 而是在训练的过程中，输出训练效果分数，比如：accuracy，f1score，recall-score等
        self.accuracy = CategoricalAccuracy()
        
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # 这个非常重要，一个batch中不同文本有不同的长度，故需要获取mask来指定参数更新梯度
        mask = get_text_field_mask(sentence)
        
        # 将sentence数据映射到词向量。
        """
        注意，这里 sentence 类型是 Dict[str,torch.Tensor] 
        
        比如token_indexers设置了 word,charaters 两个不同的token_indexer，则此处的sentence也会有这两个key，
        
        并交给text_field_embedding（也包含word，charaters这两个TokenEmbedder）映射成词向量。
        """
        embeddings = self.word_embeddings(sentence)
        
        # seq2seq_encoder
        encoder_out = self.encoder(embeddings, mask)
        
        # shape : (batch_size, sequence_length , label_size)
        tag_logits = self.hidden2tag(encoder_out)
        
        
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


reader = PosDatasetReader()
train_dataset = reader.read(cached_path('./data/train.txt'))
validation_dataset = reader.read(cached_path('./data/validation.txt'))

# 手动构造Vocabulary
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

# 自定义超参数
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

# "tokens" 是需要和DataserReader中的token_indexers的 key 保持一致
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1


# 自定义优化器
optimizer = optim.Adagrad(model.parameters())
# 自定义数据迭代方式
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])

# 现在回想一下，token_indexers 要想将token映射到index，那没有vocabulary如何映射呢？
# 这里就是给iterator设置vocab。
# 在iterator对数据进行组装的时候，会调用token_indexer函数并将vocab传递过去，此时token_indexer才会接触到vocab。
iterator.index_with(vocab)

# 这里的方法就很类似于keras的compile函数了
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)
trainer.train()