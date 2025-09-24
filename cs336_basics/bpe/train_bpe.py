import os
import heapq
import regex
from typing import Pattern
from collections import defaultdict, OrderedDict
from tests.common import FIXTURES_PATH
from multiprocessing import Pool
from functools import partial
from functools import total_ordering
from .pretokenization_example import find_chunk_boundaries


NUM_PROCESSES = 4


# Use doubly linked list to find neighbour bytes O(1) in a pre-token
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
    """
    Train bpe tokenizer with merge optimization
    """
    # Put special tokens into vocabulary
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    cur_token_id = len(vocab)
    
    # Put 0-255 byte into vocabulary
    vocab.update({cur_token_id + i: bytes([i]) for i in range(256)})
    cur_token_id = len(vocab)

    # Currently, vocab size = len(sepcial_tokens) + 256 

    # Read file content as byte stream
    with open(input_path, 'rb') as f:
        # Split the text into at most NUM_PROCESSES chunks with <|endoftext|> be the boundaries
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

        # Combine all special tokens separated by | to construct regex expression
        # Note that special tokens should be sorted by length in decreasing order
        # Because long special tokens maybe includes short ones which leads to the long ones never get matched
        if special_tokens:
            split_re = regex.compile(
                "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))

        # Boundary start:end as the dynamic parameter of workers
        jobs = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        # Use closure to capture the common parameters of workers
        # Workers run pre_tokenization
        worker = partial(pre_tokenization, input_path, split_re)
        
        with Pool(processes=NUM_PROCESSES) as pool:
            # Use multi-process workers to run pre_tokenization jobs and store results in partial_counts
            partial_counts: list[defaultdict[bytes, int]] = pool.starmap(worker, jobs)
            pretokens_count = defaultdict(int)
            # Aggregate pretoken count from all processes
            for local_counts in partial_counts:
                for token, count in local_counts.items():
                    pretokens_count[token] += count

            new_vocab, merge = bpe_merge(pretokens_count, cur_token_id, vocab_size - len(vocab))
            vocab.update(new_vocab)

        return vocab, merge


def pre_tokenization(
        input_path: str | os.PathLike, 
        split_re: Pattern[str], 
        start: int, 
        end: int) -> defaultdict[bytes, int]:
    
    with open(input_path, 'rb') as f:
        # Read the chunk assigned to current worker
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Split by special tokens (remove them) before pre-tokenize
        if split_re:
            chunks = [segment for segment in split_re.split(chunk) if segment]
        else:
            chunks = [chunk]

        # GPT-2 pre-tokenization regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        pretokens_count = defaultdict(int)

        # Pre-tokenize by chunk
        for chunk in chunks:
            # Use regex.finditer to stream the match result
            for match in regex.finditer(PAT, chunk):
                token = match.group(0).encode("utf-8")
                pretokens_count[token] += 1

        return pretokens_count


def bpe_merge(
        pretokens_count: defaultdict[bytes, int], 
        cur_token_id: int, 
        steps: int) -> tuple[dict[int, bytes], tuple[bytes, bytes]]:
    
    # Turn pre-tokens to one byte list 
    pretoken_bytelist_cnt: list[tuple[list[bytes], int]] = [([bytes([b]) for b in token], cnt) for token, cnt in pretokens_count.items()]
    pair_count: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_record: defaultdict[tuple[bytes, bytes], OrderedDict[tuple[ListNode, ListNode], int]] = defaultdict(OrderedDict)
    vocab, merge = {}, []

    for bytelist, cnt in pretoken_bytelist_cnt:
        last = None  # last node
        # From 0 to n - 2, group adjacent bytes like (0, 1), (1, 2), ..., (n-2, n-1)
        for j in range(len(bytelist) - 1):
            pair = (bytelist[j], bytelist[j + 1])
            pair_count[pair] += cnt

            # Link list
            node1 = ListNode(bytelist[j]) if last is None else last
            node2 = ListNode(bytelist[j + 1])
            node1.right = node2
            node2.left = node1
            last = node2
            
            # (bytes, bytes) to their (node1, node2)
            # Note that for all the same pairs in a single pre-token, their (node1, node2) must be stored in left-to-right order 
            # by using OrderedDict. Besides that, maintain the order during incremental update when merge bytes
            pair_record[pair][(node1, node2)] = cnt

    # Merge steps times, but be careful the pre-tokens may have no pairs to merge anymore before reach the limit
    for _ in range(steps):
        target_pair, max_cnt = None, 0
        # Get the most frequent pair, when tie choose the lexigraphical greatest pair
        # Scan the list to get the result in O(N)
        for pair, cnt in pair_count.items():
            if cnt > max_cnt:
                max_cnt, target_pair = cnt, pair
            elif cnt == max_cnt and pair > target_pair:
                target_pair = pair

        # Be careful the pre-tokens may have no pairs to merge anymore before reach the limit
        if target_pair is None:
            break

        deleted: set[ListNode] = set()
        pair = target_pair

        # Copy the items to for loop because it doesn't allow removing the dict item during iterating it
        kvs: OrderedDict[tuple[ListNode, ListNode], int] = list(pair_record[pair].items())
        for node_tup, cnt in kvs:
            node1, node2 = node_tup

            # e.g. pre-token = ['a', 'a', 'a', 'a'], pairs = ['a0a1', 'a1a2', 'a2a3'], target_pair = 'aa'
            # Step1: merge a0a1, pre-token = ['a0a1', 'a2', 'a3']
            # Step2: merge a1a2, due to a1 has been merged with a0, continue to merge a2a3, pre-token = ['a0a1', 'a2a3'] 
            if node1 in deleted:
                continue
            
            # Update incrementally
            # e.g. pre-token = ['x', 'a', 'a', 'a', 'x'], target_pair = 'aa'

            # old_pairs = ['xa', 'aa', 'aa', 'ax'], remove 'xa'(left) + 'aa'(target) + 'aa'(right)
            # new_pairs = ['xaa', 'aaa', 'ax'],     add 'xaa'(left) + 'aaa'(right)

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

            # Merge node2 to node1, then remove node2
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

        vocab[cur_token_id] = target_pair[0] + target_pair[1]
        cur_token_id += 1
        merge.append(target_pair)

    return vocab, merge


def bpe_merge_pq(pretokens_count, cur_token_id, steps) -> tuple[dict[int, bytes], tuple[bytes, bytes]]:
    """
    Instead of scan the pair_record to get the most frequent pair in O(N), use a lazy delete priority queue to get it in O(logN)
    """

    pretoken_bytelist_cnt: list[tuple[list[bytes], int]] = [([bytes([b]) for b in token], cnt) for token, cnt in pretokens_count.items()]
    pair_count: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_record: defaultdict[tuple[bytes, bytes], OrderedDict[tuple[ListNode, ListNode], int]] = defaultdict(OrderedDict)
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
        target_pair = None
        # Lazy delete priority queue, delete the poped pair unless its count is up-to-date
        # Another way is to use a version number for each pair
        while True:
            cnt, pairOrder = heapq.heappop(pq)
            cnt, pair = -cnt, pairOrder.pair
            target_pair = pair
            if cnt == pair_count[pair]:
                break

        if target_pair is None:
            break

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

        vocab[cur_token_id] = target_pair[0] + target_pair[1]
        cur_token_id += 1
        merge.append(target_pair)

    return vocab, merge


# vocab, merges = train_bpe(
#     FIXTURES_PATH / "test.txt",
#     vocab_size=256+3,
#     special_tokens = ["<|endoftext|>"],
# )

# print(vocab)
# print(merges)
