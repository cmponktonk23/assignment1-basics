import json
import regex
from collections import defaultdict, OrderedDict
from typing import Iterable, Iterator
from .train_bpe import ListNode


class BPETokenizer:
    """
    Use vocab and merges to encode and decode text to/from token IDs.
    """

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.INF = len(self.merges) + 1

        self.vocab_r: dict[bytes, int] = {token: idx for idx, token in self.vocab.items()}
        self.pair_pos: dict[bytes, int] = {pair: i for i, pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, encoding="utf-8") as vf:
            raw = json.load(vf)
            vocab = {int(idx): token.encode("utf-8") for idx, token in raw.items()}
        
        with open(merges_filepath, encoding="utf-8") as mf:
            merges = [
                (left.encode("utf-8"), right.encode("utf-8"))
                for line in mf
                for left, right in line.strip().split()
            ]
    
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        """
        pretokens_pos, total, special_token_pos = self.pre_tokenization(text)
        result = [None for _ in range(total)]  # placeholder

        # Fill special tokens' IDs into result
        for pos, idx in special_token_pos:
            result[pos] = [idx]
        
        pretoken_bytelist_pos: list[tuple[list[bytes], list[int]]] = [([bytes([b]) for b in token], pos_list) for token, pos_list in pretokens_pos.items()]

        pair_record: defaultdict[tuple[bytes, bytes], OrderedDict[tuple[ListNode, ListNode], int]] = defaultdict(OrderedDict)
        pretoken_head: dict[ListNode, list[int]] = {}

        for bytelist, pos_list in pretoken_bytelist_pos:
            last = None
            for j in range(len(bytelist) - 1):
                pair = (bytelist[j], bytelist[j + 1])

                # Link list
                node1 = ListNode(bytelist[j]) if last is None else last
                node2 = ListNode(bytelist[j + 1])
                
                # Record the head node of each pre-token's linked list
                if last is None:
                    pretoken_head[node1] = pos_list
        
                node1.right = node2
                node2.left = node1
                last = node2
                
                pair_record[pair][(node1, node2)] = 1

            # Only one byte in pre-token, store head node of the pre-token
            if last is None and len(bytelist) == 1:
                node = ListNode(bytelist[0])
                pretoken_head[node] = pos_list

        while True:
            target_pair, min_pos = None, self.INF
            # Find the most frequent pair, greatest lexigraphical one for tie
            for pair in pair_record.keys():
                if pair in self.pair_pos:
                    pos = self.pair_pos[pair]
                    if pos < min_pos:
                        min_pos, target_pair = pos, pair
            
            if target_pair is None:
                break

            deleted: set[ListNode] = set()
            pair = target_pair

            # Copy the items to for loop because it doesn't allow removing the dict item during iterating it
            kvs = list(pair_record[target_pair].items())
            for node_tup, pos_list in kvs:
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
                    
                del pair_record[pair][(node1, node2)]
                if len(pair_record[pair]) == 0:
                    del pair_record[pair]

                if node1.left:
                    left = (node1.left.b, node1.b)
                    del pair_record[left][(node1.left, node1)]
                    if len(pair_record[left]) == 0:
                        del pair_record[left]

                if node2.right:
                    right = (node2.b, node2.right.b)
                    del pair_record[right][(node2, node2.right)]
                    if len(pair_record[right]) == 0:
                        del pair_record[right]

                # Merge node2 to node1, then remove node2
                node1.b += node2.b
                node1.right = node2.right
                if node1.right: node1.right.left = node1
                deleted.add(node2)

                if node1.left:
                    left = (node1.left.b, node1.b)
                    pair_record[left][(node1.left, node1)] = 1

                if node1.right:
                    right = (node1.b, node1.right.b)
                    pair_record[right][(node1, node1.right)] = 1
        
        for head, pos_list in pretoken_head.items():
            # Walk along the linked list of each pre-token to construct the token IDs list 
            token_ids = []
            cur = head
            while cur:
                token_ids.append(self.vocab_r[cur.b])
                cur = cur.right
            
            # Fill each pre-token's token IDs list into placeholder 
            for pos in pos_list:
                result[pos] = token_ids
        
        # Flatten the 2d list to 1d
        ret = []
        for l in result:
            ret.extend(l)

        return ret


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode text by stream.
        """
        for line in iterable:
            ret = self.encode(line)
            for idx in ret:
                yield idx


    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        """
        bytes_list = [self.vocab[token_id] for token_id in ids]
        return b"".join(bytes_list).decode("utf-8", errors='replace')  # for bytes can't decode to Unicode, replace with Unicode U+FFFD


    def pre_tokenization(self, text: str) -> tuple[dict[bytes, list[int]], int]:
        # Different with train bpe tokenizer, need to keep the special tokens, use finditer but not split
        if self.special_tokens:
            pattern = regex.compile(
                    "|".join(regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            last = 0
            chunks = []
            for match in pattern.finditer(text):
                # Use special tokens to split text to chunks, keep each special token as a standalone chunk
                # Do not allow append chunk for the case '<|endoftext|><|endoftext|>'
                if match.start() > last:
                    chunks.append(text[last:match.start()])
                chunks.append(match.group(0))
                last = match.end()
            if last < len(text):
                chunks.append(text[last:])
        else:
            chunks = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        pretokens_pos: defaultdict[bytes, list[int]] = defaultdict(list)

        # Pre-tokenize by chunk
        total = 0
        special_token_set = set(self.special_tokens) if self.special_tokens else set()
        special_token_pos: tuple[int, int] = []  # [(pre-token position in text, token_id)]
        for chunk in chunks:
            # Each special token is a standalone chunk
            if chunk in special_token_set:
                special_token_pos.append((total, self.vocab_r[chunk.encode('utf-8')]))
                total += 1
                continue
            for match in regex.finditer(PAT, chunk):
                token = match.group(0).encode("utf-8")
                pretokens_pos[token].append(total)
                total += 1

        return pretokens_pos, total, special_token_pos
