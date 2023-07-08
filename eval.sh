CUDA_VISIBLE_DEVICES=0 python src/finetune_lora.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_eval \
    --dataset_dir data \
    --dataset cpp_interview_test \
    --checkpoint_dir baichuan_lora_checkpoint \
    --output_dir baichuan_lora_eval_result \
    --per_device_eval_batch_size 4 \
    --max_samples 100 \
    --predict_with_generate
    