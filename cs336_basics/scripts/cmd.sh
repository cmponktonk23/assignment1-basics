uv run cs336_basics/scripts/train_bpe.py \
    --input_path "data/TinyStoriesV2-GPT4-train.txt" \
    --out_dir "data/"


uv run cs336_basics/scripts/tokenization.py \
    --num_processes 8 \
    --input_path "data/TinyStoriesV2-GPT4-train.txt" \
    --out_path "data/TinyStoriesV2-GPT4-train" \
    --vocab_path "data/vocab.json" \
    --merges_path "data/merges.txt"


uv run cs336_basics/scripts/tokenization.py \
    --num_processes 8 \
    --input_path "data/TinyStoriesV2-GPT4-valid.txt" \
    --out_path "data/TinyStoriesV2-GPT4-valid" \
    --vocab_path "data/vocab.json" \
    --merges_path "data/merges.txt"


uv run cs336_basics/scripts/train.py \
    --wandb_mode "online" \
    --train_dataset_path "data/TinyStoriesV2-GPT4-train" \
    --train_meta_path "data/TinyStoriesV2-GPT4-train.meta.json" \
    --val_dataset_path "data/TinyStoriesV2-GPT4-valid" \
    --val_meta_path "data/TinyStoriesV2-GPT4-valid.meta.json" \
    --ckpt_path "data/ckpt"


uv run cs336_basics/scripts/decode.py \
    --prompt "Hi, my name is" \
    --max_token_num 20 \
    --vocab_path "cs336_basics/scripts/data/vocab.json" \
    --merges_path "cs336_basics/scripts/data/merges.txt" \
    --ckpt_path "cs336_basics/scripts/data/ckpt" \

