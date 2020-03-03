"""
针对中文分词定制一些简单的分词器

以便开箱即用
"""
from allennlp.data.tokenizers import Tokenizer
from allennlp.data import Token
from typing import List
from overrides import overrides
import jieba


@Tokenizer.register("jieba")
class JiebaTokenizer(Tokenizer):
    """
    jieba分词器
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        重写分词函数

        Returns
        -------
        tokens : ``List[Token]``
        """
        tokens = jieba.lcut(text)
        return [Token(token) for token in tokens]


@Tokenizer.register("pku_seg")
class PkuSegTokenizer(Tokenizer):
    """
    使用PkuSegment工具进行中文分词

    参考链接：https://github.com/lancopku/pkuseg-python
    """
    def __init__(self,domain_name = "default"):
        self.seg = pkuseg.pkuseg(model_name = domain_name)
    
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        重写分词函数

        Returns
        -------
        tokens : ``List[Token]``
        """
        tokens = self.seg.cut(text)
        return [Token(token) for token in tokens]