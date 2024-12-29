# DL_Transformer

- 2024년 4학년 2학기 딥러닝 기반 데이터 분석 Transformer 아카이빙 레포지토리입니다.

<br/><br/>

## 1. Development Environment Assign
- Python version: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)] 
- PyTorch version: 1.8.1+cpu 
- TorchText version: 0.9.1 
- CUDA version: 11.8
- Numpy version : 1.22.0

<br/><br/>

## 2. logic
### 2.1 Util
2.1.1 tokenizer

import spacy

class Tokenizer:

    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
