uv run cs336_basics/scripts/train_bpe.py \
    --num_processes 8 \
    --show_progress \
    --input_path "data/OpenWebText/owt_train.txt" \
    --out_dir "data/OpenWebText/" \
    --vocab_size 32000


uv run cs336_basics/scripts/tokenization.py \
    --num_processes 8 \
    --input_path "data/OpenWebText/owt_train.txt" \
    --out_path "data/OpenWebText/owt_train" \
    --vocab_path "data/OpenWebText/vocab.json" \
    --merges_path "data/OpenWebText/merges.txt"


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
    --prompt "Once upon a time" \
    --max_token_num 5000 \
    --vocab_path "data/vocab.json" \
    --merges_path "data/merges.txt" \
    --ckpt_path "data/ckpt"

