import os
import heapq
import regex
from collections import defaultdict, OrderedDict
from tests.common import FIXTURES_PATH
from multiprocessing import Pool
from functools import partial
from functools import total_ordering
from cs336_basics.pretokenization_example import find_chunk_boundaries


class ListNode:
    __slots__ = ("b", "left", "right")
    def __init__(self, b: bytes):
        self.b = b
        self.left, self.right = None, None


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # put special tokens into vocabulary
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    cur_id = len(vocab)
    
    # put 0-255 byte into vocabulary
    vocab.update({cur_id + i: bytes([i]) for i in range(256)})
    cur_id = len(vocab)

    with open(input_path, 'rb') as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        if special_tokens:
            split_re = regex.compile(
                "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
            )

        jobs = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        worker = partial(pre_tokenization, input_path, split_re)
        with Pool(processes=num_processes) as pool:
            partial_counts = pool.starmap(worker, jobs)
            
            pretokens_count = defaultdict(int)
            for local_counts in partial_counts:
                for token, count in local_counts.items():
                    pretokens_count[token] += count

            new_vocab, merge = bpe_merge(pretokens_count, cur_id, vocab_size - len(vocab))
            vocab.update(new_vocab)

        return vocab, merge


def pre_tokenization(input_path, split_re, start, end):
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # remove all special tokens before pre-tokenize and split by special tokens
        if split_re:
            chunks = [segment for segment in split_re.split(chunk) if segment]
        else:
            chunks = [chunk]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        pretokens_count = defaultdict(int)

        # pre-tokenize by chunk
        for chunk in chunks:
            for match in regex.finditer(PAT, chunk):
                token = match.group(0).encode("utf-8")
                pretokens_count[token] += 1

        return pretokens_count


def bpe_merge(pretokens_count, cur_id, steps) -> tuple[dict[int, bytes], tuple[bytes, bytes]]:
    pretoken_bytelist_cnt = [([bytes([b]) for b in token], cnt) for token, cnt in pretokens_count.items()]
    pair_count = defaultdict(int)
    pair_record = defaultdict(OrderedDict)
    vocab, merge = {}, []

    for bytelist, cnt in pretoken_bytelist_cnt:
        last = None
        for j in range(len(bytelist) - 1):
            pair = (bytelist[j], bytelist[j + 1])
            pair_count[pair] += cnt

            node1 = ListNode(bytelist[j]) if last is None else last
            node2 = ListNode(bytelist[j + 1])
            
            node1.right = node2
            node2.left = node1
            last = node2
            
            pair_record[pair][(node1, node2)] = cnt

    for _ in range(steps):
        target_pair, max_cnt = None, 0
        # get most frequent pair, when tie get lexigraphical greatest pair
        for pair, cnt in pair_count.items():
            if cnt > max_cnt:
                max_cnt, target_pair = cnt, pair
            elif cnt == max_cnt and pair > target_pair:
                target_pair = pair

        if target_pair:
            deleted = set()
            pair = target_pair
            kvs = list(pair_record[pair].items())
            
            for node_tup, cnt in kvs:
                node1, node2 = node_tup

                if node1 in deleted:
                    continue
                    
                pair_count[pair] -= cnt
                if pair_count[pair] == 0: del pair_count[pair]
                del pair_record[pair][(node1, node2)]

                if node1.left:
                    left = (node1.left.b, node1.b)
                    pair_count[left] -= cnt
                    if pair_count[left] == 0: del pair_count[left]
                    del pair_record[left][(node1.left, node1)]
                    
                if node2.right:
                    right = (node2.b, node2.right.b)
                    pair_count[right] -= cnt
                    if pair_count[right] == 0: del pair_count[right]
                    del pair_record[right][(node2, node2.right)]

                node1.b += node2.b
                node1.right = node2.right
                if node1.right: node1.right.left = node1
                deleted.add(node2)

                if node1.left:
                    left = (node1.left.b, node1.b)
                    pair_count[left] += cnt
                    pair_record[left][(node1.left, node1)] = cnt

                if node1.right:
                    right = (node1.b, node1.right.b)
                    pair_count[right] += cnt
                    pair_record[right][(node1, node1.right)] = cnt

            vocab[cur_id] = target_pair[0] + target_pair[1]
            cur_id += 1
            merge.append(target_pair)

    return vocab, merge


def bpe_merge_pq(pretokens_count, cur_id, steps) -> tuple[dict[int, bytes], tuple[bytes, bytes]]:
    pretoken_bytelist_cnt = [([bytes([b]) for b in token], cnt) for token, cnt in pretokens_count.items()]
    pair_count = defaultdict(int)
    pair_record = defaultdict(OrderedDict)
    vocab, merge = {}, []

    for bytelist, cnt in pretoken_bytelist_cnt:
        last = None
        for j in range(len(bytelist) - 1):
            pair = (bytelist[j], bytelist[j + 1])
            pair_count[pair] += cnt

            node1 = ListNode(bytelist[j]) if last is None else last
            node2 = ListNode(bytelist[j + 1])
            
            node1.right = node2
            node2.left = node1
            last = node2
            
            pair_record[pair][(node1, node2)] = cnt

    @total_ordering
    class PairOrder:
        __slots__ = ("pair",)
        def __init__(self, pair):
            self.pair = pair
        def __lt__(self, other):
            return self.pair > other.pair
        def __eq__(self, other):
            return self.pair == other.pair

    pq = [(-cnt, PairOrder(pair)) for pair, cnt in pair_count.items()]
    heapq.heapify(pq)

    def update_pair_count(pair, cnt):
        pair_count[pair] += cnt
        if pair_count[pair] == 0: del pair_count[pair]
        else:heapq.heappush(pq, (-pair_count[pair], PairOrder(pair)))

    for _ in range(steps):
        # get most frequent pair, when tie get lexigraphical greatest pair
        target_pair = None
        while True:
            cnt, pairOrder = heapq.heappop(pq)
            cnt, pair = -cnt, pairOrder.pair
            target_pair = pair
            if cnt == pair_count[pair]:
                break

        if target_pair:
            deleted = set()
            pair = target_pair
            kvs = list(pair_record[pair].items())
            
            for node_tup, cnt in kvs:
                node1, node2 = node_tup

                if node1 in deleted:
                    continue
                
                update_pair_count(pair, -cnt)
                
                del pair_record[pair][(node1, node2)]

                if node1.left:
                    left = (node1.left.b, node1.b)
                    update_pair_count(left, -cnt)
                    del pair_record[left][(node1.left, node1)]
                    
                if node2.right:
                    right = (node2.b, node2.right.b)
                    update_pair_count(right, -cnt)
                    del pair_record[right][(node2, node2.right)]

                node1.b += node2.b
                node1.right = node2.right
                if node1.right: node1.right.left = node1
                deleted.add(node2)

                if node1.left:
                    left = (node1.left.b, node1.b)
                    update_pair_count(left, cnt)
                    pair_record[left][(node1.left, node1)] = cnt

                if node1.right:
                    right = (node1.b, node1.right.b)
                    update_pair_count(right, cnt)
                    pair_record[right][(node1, node1.right)] = cnt

            vocab[cur_id] = target_pair[0] + target_pair[1]
            cur_id += 1
            merge.append(target_pair)

    return vocab, merge


# vocab, merges = train_bpe(
#     FIXTURES_PATH / "test.txt",
#     vocab_size=256+3,
#     special_tokens = ["<|endoftext|>"],
# )

# print(vocab)
# print(merges)
