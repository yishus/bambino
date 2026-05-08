from collections import defaultdict
import heapq
import os
from multiprocessing import Pool
import regex as re
from typing import BinaryIO

type Pair = tuple[bytes, bytes]
type FreqPair = tuple[int, Pair]


class EmptyHeapError(Exception):
    """Exception raised for when heap is empty"""

    pass


def train_from_input_path(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = [],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freq_table = pretokenize(input_path, special_tokens)

    if not freq_table:
        raise RuntimeError("No tokens found")

    pair_frequency = defaultdict(int)
    pair_to_words = defaultdict(set)
    word_components = {}
    for wf in freq_table:
        for wf_i in range(len(wf) - 1):
            pair = (wf[wf_i : wf_i + 1], wf[wf_i + 1 : wf_i + 2])
            pair_frequency[pair] += freq_table[wf]
            pair_to_words[pair].add(wf)
            word_components[wf] = [bytes([b]) for b in wf]

    heap = []
    for pair, freq in pair_frequency.items():
        heapq.heappush_max(heap, (freq, pair))

    vocab = [sp.encode() for sp in special_tokens]
    vocab += [i.to_bytes() for i in range(256)]
    merges = []

    while len(vocab) < vocab_size and len(heap):
        try:
            freq, pair = lazy_eval_freq_heap_max(heap, pair_frequency)
        except EmptyHeapError:
            print("Empty heap")
            break

        merged = pair[0] + pair[1]
        vocab.append(merged)
        merges.append(pair)

        new_pairs = set()
        for word in pair_to_words[pair]:
            indices = pair_first_positions_in_word_components(
                word_components[word], pair
            )
            for word_i in indices:
                if word_i > 0:
                    # Find the pair consisting of the preceeding token and first half of the new pair
                    prev_pair = (word_components[word][word_i - 1], pair[0])
                    # Reduce the pair frequency due to the merge by the word frequency
                    # We do not simply set this pair frequency to zero because this prev pair can occur
                    # that is not preceeding the merging pair
                    pair_frequency[prev_pair] -= freq_table[word]

                    # The new pair consist of the precceding token and the new merged bytes
                    new_pair = (word_components[word][word_i - 1], merged)
                    pair_frequency[new_pair] += freq_table[word]

                    pair_to_words[new_pair].add(word)
                    new_pairs.add(new_pair)

                if word_i < len(word_components[word]) - 2:
                    next_pair = (pair[1], word_components[word][word_i + 2])
                    pair_frequency[next_pair] -= freq_table[word]

                    new_pair = (merged, word_components[word][word_i + 2])
                    pair_frequency[new_pair] += freq_table[word]

                    pair_to_words[new_pair].add(word)
                    new_pairs.add(new_pair)

            # Remove in reverse to prevent unintended index shifting
            for word_i in indices[::-1]:
                word_components[word][word_i] = merged
                del word_components[word][word_i + 1]
        del pair_to_words[pair]

        for np in new_pairs:
            heapq.heappush_max(heap, (pair_frequency[np], np))

    return {i: v for i, v in enumerate(vocab)}, merges


def lazy_eval_freq_heap_max(
    heap: list[FreqPair], pair_frequency: dict[Pair, int]
) -> FreqPair:
    """
    Pop the max FreqPair from heap, check against pair_frequency
    and update and push back into heap if needed
    """
    freq, pair = heapq.heappop_max(heap)
    while pair_frequency[pair] < freq:
        if pair_frequency[pair] > 0:
            heapq.heappush_max(heap, (pair_frequency[pair], pair))
        if not len(heap):
            raise EmptyHeapError
        next_freq, next_pair = heapq.heappop_max(heap)
        freq = next_freq
        pair = next_pair
    return freq, pair


def pair_first_positions_in_word_components(components: bytes, pair: Pair) -> list[int]:
    """
    Look for matching pairs in a list of components. The matching pair might occur
    more than once. Return all occurences.


    Example:
    components: [b'm', b'e', b'm', b'e']
    pair: (b'm', b'e')
    return [0, 2]
    """
    return [
        w_i
        for w_i in range(len(components) - 1)
        if components[w_i] == pair[0] and components[w_i + 1] == pair[1]
    ]


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
    num_processes: int = 4,
) -> dict[bytes, int]:
    """
    Open file and parallelize pretokenize_chunk per chunk
    """
    result = {}
    with open(input_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>", file_size
        )

        boundaries.append(file_size)
        chunk_ranges = [
            (0, boundaries[0]),
            *[(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)],
        ]
        chunks = []
        for cr_start, cr_end in chunk_ranges:
            f.seek(cr_start)
            chunks.append((f.read(cr_end - cr_start), special_tokens))
        with Pool(len(boundaries) + 1) as p:
            dictionaries = p.starmap(pretokenize_chunk, chunks)

        for d in dictionaries:
            result |= d

    return result


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    file_size: int,
) -> list[int]:
    """
    Split file into chunks by desired_num_chunks, search for split_special_token
    around boundaries.
    """
    chunk_size = file_size // desired_num_chunks
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    estimated_chunk_boundaries = [i * chunk_size for i in range(1, desired_num_chunks)]
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


def pretokenize_chunk(data: bytes, special_tokens) -> dict[bytes, int]:
    """
    - Remove special tokens from chunk without allowing merge along boundaries
    - Apply GPT-2 regex pre-tokenizer
    - Return count per pre-token
    """
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
