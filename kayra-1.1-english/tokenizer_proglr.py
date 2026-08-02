"""Word-level tokenizer for ProgLR models (trust_remote_code=True).

Matching the training preprocessing: lowercase + whitespace split.
Out-of-vocabulary words map to <unk> (id 1). Special tokens:
<pad>=0, <unk>=1, <bos>=2.
"""
import os
import re
from typing import List, Optional

from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}


class WordLevelProgLRTok(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, unk_token='<unk>', bos_token='<bos>',
                 pad_token='<pad>', eos_token=None, do_lower_case=True,
                 model_max_length=1024, **kwargs):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.vocab = {w: i for i, w in enumerate(lines)}
        self.ids_to_tokens = list(lines)
        self.do_lower_case = do_lower_case
        self._space_re = re.compile(r'\s+')
        super().__init__(unk_token=unk_token, bos_token=bos_token,
                         pad_token=pad_token, eos_token=eos_token,
                         model_max_length=model_max_length, **kwargs)

    def _tokenize(self, text: str) -> List[str]:
        if self.do_lower_case:
            text = text.lower()
        return [t for t in self._space_re.split(text.strip()) if t]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens[index]

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        path = os.path.join(save_directory,
                            (filename_prefix + '-' if filename_prefix else '') + 'vocab.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for w in self.ids_to_tokens:
                f.write(w + '\n')
        return (path,)
