import json
import os
import pickle
from collections import Counter
import re
import unicodedata

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


class CharTokenizer:

    CHARS = [SOS, EOS, PAD, ' ', '!', '"', '#', '&', "'", '(', ')', ',',
             '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
             'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def __init__(self, max_len=150):
        self.max_len = max_len
        self.vocab = self.CHARS
        self.idx2tok = {k: v for k, v in enumerate(self.vocab)}
        self.tok2idx = {v: k for k, v in self.idx2tok.items()}
        self.vocab_size = len(self.vocab)
        self.sos_idx = self.tok2idx[SOS]
        self.eos_idx = self.tok2idx[EOS]
        self.pad_idx = self.tok2idx[PAD]

    def encode(self, text: str) -> list:
        tokens = [self.tok2idx.get(c, self.pad_idx) for c in list(text)]
        seq = [self.sos_idx] + tokens + [self.eos_idx]
        seq = seq[:self.max_len]
        seq += [self.pad_idx] * (self.max_len - len(seq))
        return seq

    def decode(self, indices) -> str:
        chars = []
        for idx in indices:
            tok = self.idx2tok.get(int(idx), '')
            if tok == EOS:
                break
            if tok not in (SOS, PAD):
                chars.append(tok)
        return ''.join(chars)

    def name(self):
        return 'char'


class WordTokenizer:

    def __init__(self, max_len=35, min_freq=2):
        self.max_len = max_len
        self.min_freq = min_freq
        self.vocab = None
        self.vocab_size = None
        self.sos_idx = None
        self.eos_idx = None
        self.pad_idx = None
        self.unk_idx = None
        self._built = False

    @staticmethod
    def _tokenize(text: str) -> list:
        return text.lower().split()


    @staticmethod
    def _clean_caption(
        caption: str,
        normalize_accents=False,
        keep_numbers=True
    ):
        caption = caption.lower()

        if normalize_accents:
            caption = unicodedata.normalize('NFKD', caption)
            caption = caption.encode('ascii', 'ignore').decode('ascii')

        caption = caption.replace('-', ' ')

        if keep_numbers:
            caption = re.sub(r"[^\w\s]", "", caption, flags=re.UNICODE)
        else:
            caption = re.sub(r"[^\p{L}\s]", "", caption)

        caption = caption.replace('_', '')
        caption = re.sub(r"\s+", " ", caption).strip()

        return caption

    def build(self, captions: list):
        counter = Counter()
        for cap in captions:
            counter.update(self._tokenize(self._clean_caption(cap, normalize_accents=True)))
        words = [w for w, c in counter.most_common() if c >= self.min_freq]
        self.vocab = [PAD, SOS, EOS, UNK] + words
        self.idx2tok = {k: v for k, v in enumerate(self.vocab)}
        self.tok2idx = {v: k for k, v in self.idx2tok.items()}
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.tok2idx[PAD]
        self.sos_idx = self.tok2idx[SOS]
        self.eos_idx = self.tok2idx[EOS]
        self.unk_idx = self.tok2idx[UNK]
        self._built = True
        print(f"[WordTokenizer] vocab size: {self.vocab_size}, min_freq={self.min_freq}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        with open(path, 'rb') as f:
            tok.__dict__.update(pickle.load(f))
        return tok

    def encode(self, text: str) -> list:
        assert self._built, "Call build() first"
        tokens = [self.tok2idx.get(w, self.unk_idx) for w in self._tokenize(text)]
        seq = [self.sos_idx] + tokens + [self.eos_idx]
        seq = seq[:self.max_len]
        seq += [self.pad_idx] * (self.max_len - len(seq))
        return seq

    def decode(self, indices) -> str:
        words = []
        for idx in indices:
            tok = self.idx2tok.get(int(idx), UNK)
            if tok == EOS:
                break
            if tok not in (SOS, PAD, UNK):
                words.append(tok)
        return ' '.join(words)

    def name(self):
        return 'word'


class SubwordTokenizer:

    def __init__(self, max_len=50, vocab_size=4000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self._tokenizer = None
        self.sos_idx = None
        self.eos_idx = None
        self.pad_idx = None
        self._built = False

    def build(self, captions: list, save_path: str = None):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        tokenizer = Tokenizer(BPE(unk_token=UNK))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[PAD, SOS, EOS, UNK],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(captions, trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS} $A {EOS}",
            special_tokens=[(SOS, tokenizer.token_to_id(SOS)),
                            (EOS, tokenizer.token_to_id(EOS))],
        )
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id(PAD),
            pad_token=PAD,
            length=self.max_len,
        )
        tokenizer.enable_truncation(max_length=self.max_len)

        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_idx = tokenizer.token_to_id(PAD)
        self.sos_idx = tokenizer.token_to_id(SOS)
        self.eos_idx = tokenizer.token_to_id(EOS)
        self._built = True

        if save_path:
            tokenizer.save(save_path)
        print(f"[SubwordTokenizer] vocab size: {self.vocab_size}")

    def save(self, path):
        self._tokenizer.save(path)
        meta = {k: v for k, v in self.__dict__.items() if k != '_tokenizer'}
        with open(path + '.meta', 'w') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer
        tok = cls.__new__(cls)
        tok._tokenizer = Tokenizer.from_file(path)
        with open(path + '.meta') as f:
            meta = json.load(f)
        tok.__dict__.update(meta)
        tok._built = True
        return tok

    def encode(self, text: str) -> list:
        assert self._built, "Call build() first"
        enc = self._tokenizer.encode(text)
        return enc.ids

    def decode(self, indices) -> str:
        ids = []
        for idx in indices:
            i = int(idx)
            if i == self.eos_idx:
                break
            if i not in (self.sos_idx, self.pad_idx):
                ids.append(i)
        return self._tokenizer.decode(ids)

    def name(self):
        return 'subword'


def build_tokenizer(text_repr, ann_file, cache_dir, max_len=None):
    os.makedirs(cache_dir, exist_ok=True)

    if text_repr == 'char':
        return CharTokenizer(max_len=max_len or 150)

    with open(ann_file) as f:
        data = json.load(f)
    captions = [a['caption'] for a in data['annotations']]

    if text_repr == 'word':
        ml = max_len or 35
        cache_path = os.path.join(cache_dir, 'word_tokenizer.pkl')
        if os.path.exists(cache_path):
            tok = WordTokenizer.load(cache_path)
            tok.max_len = ml
            print(f"[WordTokenizer] loaded from {cache_path}")
            return tok
        tok = WordTokenizer(max_len=ml)
        tok.build(captions)
        tok.save(cache_path)
        return tok

    if text_repr == 'subword':
        ml = max_len or 50
        cache_path = os.path.join(cache_dir, 'subword_tokenizer.json')
        if os.path.exists(cache_path):
            tok = SubwordTokenizer.load(cache_path)
            tok.max_len = ml
            print(f"[SubwordTokenizer] loaded from {cache_path}")
            return tok
        tok = SubwordTokenizer(max_len=ml, vocab_size=4000)
        tok.build(captions, save_path=cache_path)
        return tok

    raise ValueError(f"Unknown text_repr: {text_repr}")
