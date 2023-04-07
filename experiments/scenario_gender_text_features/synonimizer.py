import random
import pymorphy2
import wget
import shutil
import numpy as np

from functools import lru_cache, reduce
from typing import List, Literal
from gensim.models import KeyedVectors
from razdel import tokenize
from pathlib import Path


model_dir = Path(__file__).parent / 'model'

class Synonimizer:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.w2v = Synonimizer.__load_w2v(model_dir)


    @lru_cache(maxsize=None)
    def get_similar(self, word: str, word_type: Literal['NOUN', 'ADJF', 'VERB']) -> str:
        normal_form = self.morph.parse(word)[0].normal_form
        try:
            syn_list = self.w2v.most_similar(normal_form + f'_{word_type}', topn=10)
            similar = np.random.choice(
                [synonim for synonim, _ in syn_list if word_type in synonim]
            ).split(f'_{word_type}')[0]
        except:
            similar = word
        return similar

    
    def synonimize_text(
        self, 
        text: str, 
        word_types: List[Literal['NOUN', 'ADJF', 'VERB']],
        word_change_prob: float,
    ) -> str:
        selected_words = []
        if len(text) == 0:
            return ''
        
        tokens = [_.text for _ in tokenize(text)]
    
        new_list = []

        for word_type in word_types:
            selected_words = self.select_by_pos(word_list=tokens, pos=word_type)
            for word in selected_words:
                if 'Name' in self.morph.parse(word)[0].tag: continue
                if np.random.rand() < word_change_prob:
                    synonym = self.get_similar(word, word_type)
                    if synonym != word:
                        new_list.append((word, self.make_similar(synonym, word)))
        return reduce(lambda s, items: s.replace(*items), new_list, text)     


    def make_similar(self, word: str, sim_to: str) -> str:
        tag = self.morph.parse(sim_to)[0].tag
        try:
            ans = self.morph.parse(word)[0].inflect(tag.grammemes).word
            if word.istitle():
                ans = ans.capitalize()
            if word.isupper():
                ans = ans.upper()
            return ans
        except:
            return word

    def is_noun(self, word: str) -> bool:
        return 'NOUN' in self.morph.tag(word)[0].grammemes


    def select_by_pos(self, word_list: List[str], pos: str) -> List[str]:
        return list(filter(lambda word: self.morph.parse(word)[0].tag.POS == pos, word_list))


    @staticmethod
    def __load_w2v(dir: Path):
        dir.mkdir(parents=True, exist_ok=True)
        model_path = dir / 'ruwikiruscorpora_upos_skipgram_300_2_2019.w2v'
        if not model_path.exists():
            wget.download('http://vectors.nlpl.eu/repository/20/182.zip', str(dir / '182.zip'))
            shutil.unpack_archive(dir / '182.zip', dir / '182', 'zip')
            shutil.copy(dir / '182' / 'model.bin', model_path)
            shutil.rmtree(dir / '182')
            (dir / '182.zip').unlink()

        w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        w2v_model.init_sims(replace=True)
        return w2v_model
