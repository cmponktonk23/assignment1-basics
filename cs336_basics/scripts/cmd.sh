uv run cs336_basics/scripts/train_bpe.py \
    --num_processes 4 \
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
    --input_path "data/OpenWebText/owt_valid.txt" \
    --out_path "data/OpenWebText/owt_valid" \
    --vocab_path "data/OpenWebText/vocab.json" \
    --merges_path "data/OpenWebText/merges.txt"


uv run cs336_basics/scripts/train.py \
    --wandb_mode "online" \
    --train_dataset_path "data/OpenWebText/owt_train" \
    --train_meta_path "data/OpenWebText/owt_train.meta.json" \
    --val_dataset_path "data/OpenWebText/owt_valid" \
    --val_meta_path "data/OpenWebText/owt_valid.meta.json" \
    --ckpt_path "data/OpenWebText/ckpt" \
    --ckpt_interval 2000 \
    --log_interval 100 \
    --batch_size 40 \
    --context_length 512 \
    --vocab_size 32000 \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12 \
    --num_steps 12000 \
    --eval_steps 20 \
    --eval_interval 3000 \
    --warmup_iters 1200 \
    --cosine_cycle_iters 12000 \
    --max_learning_rate 5e-4 \
    --min_learning_rate 1e-5 \
    --weight_decay 0.05


uv run cs336_basics/scripts/decode.py \
    --prompt "I'm a businessman!" \
    --max_token_num 400 \
    --vocab_path "data/OpenWebText/vocab.json" \
    --merges_path "data/OpenWebText/merges.txt" \
    --ckpt_path "data/OpenWebText/ckpt" \
    --context_length 512 \
    --vocab_size 32000 \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12 \


#################################################################################################

uv run cs336_basics/scripts/train_bpe.py \
    --num_processes 8 \
    --show_progress \
    --input_path "data/TinyStories/TinyStoriesV2-GPT4-train.txt" \
    --out_dir "data/TinyStories/" \
    --vocab_size 10000


uv run cs336_basics/scripts/tokenization.py \
    --num_processes 8 \
    --input_path "data/TinyStories/TinyStoriesV2-GPT4-train.txt" \
    --out_path "data/TinyStories/TinyStoriesV2-GPT4-train" \
    --vocab_path "data/TinyStories/vocab.json" \
    --merges_path "data/TinyStories/merges.txt"


uv run cs336_basics/scripts/tokenization.py \
    --num_processes 8 \
    --input_path "data/TinyStories/TinyStoriesV2-GPT4-valid.txt" \
    --out_path "data/TinyStories/TinyStoriesV2-GPT4-valid" \
    --vocab_path "data/TinyStories/vocab.json" \
    --merges_path "data/TinyStories/merges.txt"


uv run cs336_basics/scripts/train.py \
    --seed 66
    --wandb_mode "online" \
    --train_dataset_path "data/TinyStories/TinyStoriesV2-GPT4-train" \
    --train_meta_path "data/TinyStories/TinyStoriesV2-GPT4-train.meta.json" \
    --val_dataset_path "data/TinyStories/TinyStoriesV2-GPT4-valid" \
    --val_meta_path "data/TinyStories/TinyStoriesV2-GPT4-valid.meta.json" \
    --ckpt_path "data/TinyStories/ckpt" \
    --ckpt_interval 5000 \
    --log_interval 100 \
    --batch_size 32 \
    --context_length 256 \
    --vocab_size 10000 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --num_steps 40000 \
    --eval_steps 10 \
    --eval_interval 4000 \
    --warmup_iters 4000 \
    --cosine_cycle_iters 40000 \
    --max_learning_rate 3e-4 \
    --min_learning_rate 1e-5 \
    --weight_decay 0.05

uv run cs336_basics/scripts/decode.py \
    --prompt "I have no friend" \
    --max_token_num 8000 \
    --vocab_path "data/TinyStories/vocab.json" \
    --merges_path "data/TinyStories/merges.txt" \
    --ckpt_path "data/TinyStories/ckpt"


