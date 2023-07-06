CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_eval \
    --dataset cpp_interview_test \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_lora_checkpoint \
    --per_device_eval_batch_size 4 \
    --max_samples 50 \
    --predict_with_generate