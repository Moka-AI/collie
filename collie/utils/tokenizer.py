from typing import cast, List

from collie.types import Tokenizer


def infer_num_end_tokens(tokenizer: Tokenizer) -> int:
    words = ['a', 'of', '我', '的']
    for word in words:
        word_id = tokenizer.convert_tokens_to_ids(word)
        word_in_vocab = word_id != 0
        if word_in_vocab:
            ids = tokenizer(word)['input_ids']
            ids = cast(List[int], ids)
            if word_id in ids:
                return len(ids) - ids.index(word_id) - 1
    raise ValueError(f'can not infer num_end_tokens from {tokenizer.__class__.__name__}')
