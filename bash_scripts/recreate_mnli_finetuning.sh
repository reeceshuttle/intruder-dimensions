CUDA_VISIBLE_DEVICES=0; 

python finetuning.py \
    --base_model=roberta-base \
    --task=mnli \
    --method=lora \
    --tune_layers=all_linear \
    --rank=1 \
    --lora_alpha=2 \
    --learning_rate=3e-4 \
    --batch_size=16 \
    --num_epochs=5 \
    --steps_per_validate=1000 \
    --max_seq_length=512 \
    --save_result=True;

python finetuning.py \
    --base_model=roberta-base \
    --task=mnli \
    --method=lora \
    --tune_layers=all_linear \
    --rank=8 \
    --lora_alpha=16 \
    --learning_rate=1.5e-4 \
    --batch_size=16 \
    --num_epochs=5 \
    --steps_per_validate=1000 \
    --max_seq_length=512 \
    --save_result=True;

python finetuning.py \
    --base_model=roberta-base \
    --task=mnli \
    --method=full \
    --tune_layers=all_linear \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --num_epochs=5 \
    --steps_per_validate=1000 \
    --max_seq_length=512 \
    --save_result=True
