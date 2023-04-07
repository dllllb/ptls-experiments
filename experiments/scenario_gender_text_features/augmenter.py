import numpy as np
import re


class Augmenter:
    def __init__(
        self, 
        wordlist: list = None, 
        alphabet: str = None
    ):
        if alphabet is None:
            alphabet = list('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя' + ',!?-.')
        if wordlist is None:
            wordlist = list()
            
        self.alphabet = alphabet
        self.wordlist = wordlist
    
    def skip_symbol(self, text: str, prob: float = 0.5) -> str:
        if not self._toss(prob):
            return text
        idx = np.random.randint(1, len(text) - 1)
        return text[:idx] + text[idx+1:]
    
    def swap_symbols(self, text: str, prob: float = 0.5) -> str:
        if not self._toss(prob):
            return text
        idx = np.random.randint(1, len(text) - 1)
        return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    
    def replace_symbol(self, text: str, prob: float = 0.5) -> str:
        if not self._toss(prob):
            return text
        idx = np.random.randint(0, len(text))
        return text[:idx] + np.random.choice(self.alphabet) + text[idx+1:]
    
    def add_entity(self, text: str, prob: float = 0.5) -> str:
        if not self._toss(prob):
            return text
        
        positions = [m.start() for m in re.finditer(r'[^\S\r\n]', text)]
        if len(positions) == 0:
            return text
        
        pos = np.random.choice(positions)
        if self._toss(0.5):
            return text[:pos+1] + str(np.random.randint(0, 9999999)) + text[pos:]
        else:
            return text[:pos+1] + np.random.choice(self.wordlist) + text[pos:]
    
    def word_abb(self, text: str, prob: float = 0.5) -> str:
        if not self._toss(prob):
            return text
        
        words = text.split()
        indices = [i for i, word in enumerate(words) if len(word) > 3]
        if len(indices) == 0:
            return text
        
        idx = np.random.choice(indices)
        
        if self._toss(0.5):
            words[idx] = words[idx][:3] + '.'
        else:
            words[idx] = words[idx][:3] + '-' + words[idx][-1]
        
        return ' '.join(words)
        
    def _toss(self, prob) -> bool:
        return np.random.rand() <= prob
