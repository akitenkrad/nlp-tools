from typing import List
from abc import ABC, abstractmethod
from nltk import word_tokenize

class Tokenizer(ABC):
    
    @abstractmethod
    def preprocess(self, text):
        pass
    
    @abstractmethod
    def tokenize(self, text):
        pass

class WordTokenizer(Tokenizer):
    
    def __init__(self, pad='<pad>', max_sent_len=-1):
        self.PAD = pad
        self.max_sent_len = max_sent_len
        
    def preprocess(self, text) -> str:
        # lowercase
        text = text.lower()

        return text
    
    def tokenize(self, text:str, disable_max_len=False) -> List[str]:
        '''tokenize a sentence
        
        Args:
            disable_max_len: if True, ignore the "max_sent_len" setting
            
        Return:
            [t_1, t_2, t_3, ...]
        '''
        text = self.preprocess(text)
        
        # remove punctuation
        # words = [word for word in word_tokenize(text) if word not in list(string.punctuation)]
        
        words = [word for word in word_tokenize(text)]
        
        # pad sentence
        if self.max_sent_len > 0 and disable_max_len == False:
            words = words + [self.PAD] * self.max_sent_len
            words = words[:self.max_sent_len]
        
        return words

class CharTokenizer(Tokenizer):
    
    def __init__(self, pad='<pad>', max_sent_len=-1, max_word_len=-1):
        self.PAD = pad
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        
    def preprocess(self, text) -> str:
        # lowercase
        text = text.lower()
        
        return text
    
    def tokenize(self, text:str) -> List[List[str]]:
        '''tokenize a sentence
        
        Return:
            [[c_1_1, c_1_2, ...], [c_2_1, c_2_2, ...], ...]
        '''
        text = self.preprocess(text)
        
        # remove punctuation
        # words = [word for word in word_tokenize(text) if word not in list(string.punctuation)]
        
        words = [word for word in word_tokenize(text)]
           
        chars = []
        for word in words:
            word = list(word)
            
            # pad words
            if self.max_word_len > 0:
                word = word + [self.PAD] * self.max_word_len
                word = word[:self.max_word_len]
                
            chars.append(word)
            
        # pad sentence
        if self.max_sent_len > 0:
            chars = chars + [[self.PAD] * self.max_word_len] * self.max_sent_len
            chars = chars[:self.max_sent_len]
        
        return chars
    