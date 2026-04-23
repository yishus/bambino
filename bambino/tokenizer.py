from collections import defaultdict
import heapq
import os
from multiprocessing import Pool
import regex as re
from typing import BinaryIO, Self


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = [],
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def train_from_input_path(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] = [],
        save_to_file: bool = False,
    ) -> Self:
        pretokenized_tokens = cls._pretokenize(input_path, special_tokens)

        pair_frequency = defaultdict(int)
        pair_to_words = defaultdict(set)
        word_components = {}
        for pt in pretokenized_tokens:
            for ptb_i in range(len(pt) - 1):
                pair = (pt[ptb_i : ptb_i + 1], pt[ptb_i + 1 : ptb_i + 2])
                pair_frequency[pair] += pretokenized_tokens[pt]
                pair_to_words[pair].add(pt)
            word_components[pt] = [bytes([b]) for b in pt]

        heap = []
        for pair, freq in pair_frequency.items():
            heapq.heappush_max(heap, (freq, pair))

        vocab = [sp.encode() for sp in special_tokens]
        vocab += [i.to_bytes() for i in range(256)]
        merges = []

        while len(vocab) < vocab_size and len(heap):
            empty_heap = False
            freq, pair = heapq.heappop_max(heap)
            while pair_frequency[pair] < freq:
                if pair_frequency[pair] > 0:
                    heapq.heappush_max(heap, (pair_frequency[pair], pair))
                if not len(heap):
                    empty_heap = True
                    break
                next_freq, next_pair = heapq.heappop_max(heap)
                freq = next_freq
                pair = next_pair
            if empty_heap:
                break
            merged = pair[0] + pair[1]
            vocab.append(merged)
            merges.append(pair)
            new_pairs = set()
            for word in pair_to_words[pair]:
                indices = [
                    w_i
                    for w_i in range(len(word_components[word]) - 1)
                    if word_components[word][w_i] == pair[0]
                    and word_components[word][w_i + 1] == pair[1]
                ]
                for word_i in indices:
                    if word_i > 0:
                        prev_pair = (word_components[word][word_i - 1], pair[0])
                        pair_frequency[prev_pair] -= pretokenized_tokens[word]

                        new_pair = (word_components[word][word_i - 1], merged)
                        pair_frequency[new_pair] += pretokenized_tokens[word]

                        pair_to_words[new_pair].add(word)
                        new_pairs.add(new_pair)

                    if word_i < len(word_components[word]) - 2:
                        next_pair = (pair[1], word_components[word][word_i + 2])
                        pair_frequency[next_pair] -= pretokenized_tokens[word]

                        new_pair = (merged, word_components[word][word_i + 2])
                        pair_frequency[new_pair] += pretokenized_tokens[word]

                        pair_to_words[new_pair].add(word)
                        new_pairs.add(new_pair)

                # Remove in reverse to prevent unintended index shifting
                for word_i in indices[::-1]:
                    word_components[word][word_i] = merged
                    del word_components[word][word_i + 1]
            del pair_to_words[pair]

            for np in new_pairs:
                heapq.heappush_max(heap, (pair_frequency[np], np))

        return cls({i: v for i, v in enumerate(vocab)}, merges, special_tokens)

    @staticmethod
    def _pretokenize(
        input_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
        num_processes: int = 4,
    ):
        with open(input_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            boundaries = BPETokenizer._find_chunk_boundaries(
                f, num_processes, b"<|endoftext|>", file_size
            )

            boundaries.append(file_size)
            chunk_ranges = [
                (0, boundaries[0]),
                *[
                    (boundaries[i], boundaries[i + 1])
                    for i in range(len(boundaries) - 1)
                ],
            ]
            chunks = []
            for cr_start, cr_end in chunk_ranges:
                f.seek(cr_start)
                chunks.append((f.read(cr_end - cr_start), special_tokens))
            with Pool(len(boundaries) + 1) as p:
                dictionaries = p.starmap(BPETokenizer._pretokenize_chunk, chunks)

            result = {}
            for d in dictionaries:
                result |= d

            return result

    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
        file_size: int,
    ) -> list[int]:
        chunk_size = file_size // desired_num_chunks
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        estimated_chunk_boundaries = [
            i * chunk_size for i in range(1, desired_num_chunks)
        ]
        chunk_boundaries = []

        reached_eof = False
        for initial_position in estimated_chunk_boundaries:
            if reached_eof:
                break

            file.seek(initial_position)

            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # Reached EOF
                if mini_chunk == b"":
                    reached_eof = True
                    break

                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries.append(initial_position + found_at)
                    break
                initial_position += mini_chunk_size

        return chunk_boundaries

    @staticmethod
    def _pretokenize_chunk(data: bytes, special_tokens) -> dict[str, int]:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""

        words = data.decode("utf-8", errors="ignore")
        chunks_without_special_tokens = [words]
        if special_tokens:
            chunks_without_special_tokens = [
                s for s in re.split(r"\|".join(special_tokens), words) if s
            ]

        tokens_count = defaultdict(int)
        for chunk in chunks_without_special_tokens:
            matches = re.finditer(pattern, chunk)
            for match in matches:
                tokens_count[match.group().encode()] += 1

        return tokens_count
