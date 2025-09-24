import json
import regex
from collections import defaultdict, OrderedDict
from typing import Iterable, Iterator
from .train_bpe import ListNode


class BPETokenizer:

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        super().__init__()

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.INF = len(self.merges) + 1

        self.vocab_r = {token: idx for idx, token in self.vocab.items()}
        self.pair_pos = {pair: i for i, pair in enumerate(self.merges)}

        # print(self.vocab)
        # print(self.merges)
        # print(self.special_tokens)

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
        pretokens_pos, total, special_token_pos = self.pre_tokenization(text)
        result = [None for _ in range(total)]
        for pos, idx in special_token_pos:
            result[pos] = [idx]
        pretoken_bytelist_pos = [([bytes([b]) for b in token], pos_list) for token, pos_list in pretokens_pos.items()]

        pair_record = defaultdict(OrderedDict)
        pretoken_head = {}

        for bytelist, pos_list in pretoken_bytelist_pos:
            last = None

            for j in range(len(bytelist) - 1):
                pair = (bytelist[j], bytelist[j + 1])

                node1 = ListNode(bytelist[j]) if last is None else last
                node2 = ListNode(bytelist[j + 1])

                if last is None:
                    pretoken_head[node1] = pos_list
                
                node1.right = node2
                node2.left = node1
                last = node2
                
                pair_record[pair][(node1, node2)] = 1

            # only one byte in pre-token
            if last is None and len(bytelist) == 1:
                node = ListNode(bytelist[0])
                pretoken_head[node] = pos_list

        while True:
            target_pair, min_pos = None, self.INF
            for pair in pair_record.keys():
                if pair in self.pair_pos:
                    pos = self.pair_pos[pair]
                    if pos < min_pos:
                        min_pos, target_pair = pos, pair
            
            if target_pair is None:
                break

            deleted = set()
            pair = target_pair
            kvs = list(pair_record[target_pair].items())

            for node_tup, pos_list in kvs:
                node1, node2 = node_tup

                if node1 in deleted:
                    continue
                    
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
            ans = []
            cur = head
            while cur:
                ans.append(self.vocab_r[cur.b])
                cur = cur.right
            for pos in pos_list:
                result[pos] = ans
        
        ret = []
        for l in result:
            ret.extend(l)

        return ret


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            ret = self.encode(line)
            for idx in ret:
                yield idx


    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[idx] for idx in ids]
        return b"".join(bytes_list).decode("utf-8", errors='replace')


    def pre_tokenization(self, text: str) -> tuple[dict[bytes, list[int]], int]:
        if self.special_tokens:
            pattern = regex.compile(
                    "|".join(regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            
            last = 0
            chunks = []
            for match in pattern.finditer(text):
                if match.start() > last:
                    chunks.append(text[last:match.start()])
                chunks.append(match.group(0))
                last = match.end()
            if last < len(text):
                chunks.append(text[last:])
        else:
            chunks = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        pretokens_pos = defaultdict(list)

        # pre-tokenize by chunk
        total = 0
        special_token_set = set(self.special_tokens) if self.special_tokens else set()
        ret = []
        for chunk in chunks:
            if chunk in special_token_set:
                ret.append((total, self.vocab_r[chunk.encode('utf-8')]))
                total += 1
                continue
            for match in regex.finditer(PAT, chunk):
                token = match.group(0).encode("utf-8")
                pretokens_pos[token].append(total)
                total += 1

        return pretokens_pos, total, ret
