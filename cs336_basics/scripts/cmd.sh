uv run cs336_basics/scripts/train_bpe.py \
    --input_path "cs336_basics/scripts/data/tinystories_sample_5M.txt" \
    --out_dir "cs336_basics/scripts/data/" \


uv run cs336_basics/scripts/tokenization.py \
    --input_path "cs336_basics/scripts/data/tinystories_sample_5M.txt" \
    --train_out_path "cs336_basics/scripts/data/tinystories_train" \
    --val_out_path "cs336_basics/scripts/data/tinystories_val" \
    --vocab_path "cs336_basics/scripts/data/vocab.json" \
    --merges_path "cs336_basics/scripts/data/merges.txt"


uv run cs336_basics/scripts/train.py \
    --train_dataset_path "cs336_basics/scripts/data/tinystories_train" \
    --train_meta_path "cs336_basics/scripts/data/tinystories_train.meta.json" \
    --val_dataset_path "cs336_basics/scripts/data/tinystories_val" \
    --val_meta_path "cs336_basics/scripts/data/tinystories_val.meta.json" \
    --ckpt_path "cs336_basics/scripts/data/ckpt" \
    --num_steps 10 \
    --eval_steps 200 \
    --warmup_iters 800 \
    --cosine_cycle_iters 1000 \
    --context_length 64 \
    --d_model 128 \
    --d_ff 336 \
    --log_interval 1 \
    --eval_steps 3


uv run cs336_basics/scripts/decode.py \
    --prompt "Hi, my name is" \
    --max_token_num 20 \
    --vocab_path "cs336_basics/scripts/data/vocab.json" \
    --merges_path "cs336_basics/scripts/data/merges.txt" \
    --ckpt_path "cs336_basics/scripts/data/ckpt" \
    --context_length 64 \
    --d_model 128 \
    --d_ff 336
