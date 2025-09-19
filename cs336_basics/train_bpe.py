from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

vocab, merges = run_train_bpe(
    FIXTURES_PATH / "test.txt",
    vocab_size=256+3,
    special_tokens = ["<|endoftext|>"],
)

print(vocab)
print(merges)
