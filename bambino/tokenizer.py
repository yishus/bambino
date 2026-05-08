import json
import os
from typing import Self

import regex as re

GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        self.rx = re.compile(GPT2_PRETOKENIZE_PATTERN)

        if self.special_tokens:
            for st in self.special_tokens:
                token_bytes = st.encode()
                # Add special tokens to vocab if not already in
                if token_bytes not in self.byte_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.byte_to_id[token_bytes] = new_id

            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile(
                "|".join(re.escape(s) for s in sorted_special)
            )
        else:
            self._special_re = None

        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        self.encoded_cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] = [],
    ) -> Self:
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
            for key in vocab:
                vocab[key] = vocab[key].encode()

        with open(merges_filepath, encoding="utf-8") as f:
            tokens = [line.rstrip().split(" ") for line in f]
            merges = [(t[0].encode(), t[1].encode()) for t in tokens]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        res = []

        if self._special_re:
            last = 0
            for match in self._special_re.finditer(text):
                if last < match.start():
                    res += self._encode_str(text[last : match.start()])
                res.append(self.byte_to_id[match.group().encode()])
                last = match.end()
            if last < len(text):
                res += self._encode_str(text[last:])
        else:
            res += self._encode_str(text)

        return res

    def _encode_str(self, text: str) -> list[int]:
        res = []
        matches = self.rx.finditer(text)
        for match in matches:
            pretoken = match.group()
            if not pretoken:
                continue
            if pretoken in self.encoded_cache:
                res += self.encoded_cache[pretoken]
            else:
                pt_bytes = pretoken.encode()
                components = [pt_bytes[i : i + 1] for i in range(len(pt_bytes))]
                if len(components) == 1:
                    ids = [self.byte_to_id[components[0]]]
                    res += ids
                    self.encoded_cache[pretoken] = ids
                    continue

                while True:
                    best_rank = None
                    best_pair = None
                    for i in range(len(components) - 1):
                        pair = (components[i], components[i + 1])
                        if pair in self.merge_ranks:
                            if not best_rank or best_rank > self.merge_ranks[pair]:
                                best_rank = self.merge_ranks[pair]
                                best_pair = pair

                    if not best_pair:
                        break

                    indices = [
                        i
                        for i in range(len(components) - 1)
                        if components[i] == best_pair[0]
                        and components[i + 1] == best_pair[1]
                    ]

                    for i in indices:
                        components[i] = best_pair[0] + best_pair[1]
                        del components[i + 1]

                res += [self.byte_to_id[c] for c in components]

        return res
