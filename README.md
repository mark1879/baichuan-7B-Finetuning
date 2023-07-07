
## 介绍
本项目使用 **baichuan-7B** 的预训练模型进行微调，目标是训练出一个能够帮助开发者备战面试的大模型，目前收集了 1000+ 条 C++ 面试相关的训练数据（数据来源：个人面试经验、互联网、GPT3.5等），欢迎开源爱好者增加其它 IT 技术领域的面试数据集。


## 环境准备
#### 1. 下载本项目
```sh
git clone https://github.com/mark1879/baichuan-7B-Finetuning.git
```

#### 2. 下载 baichuan-7B 预训练模型
```sh
# 在根目录下新建 baichuan-inc 保存大模型
mkdir baichuan-inc
cd baichuan-inc

git lfs install
git clone https://huggingface.co/baichuan-inc/baichuan-7B
```

##### 3. 安装 python 库
```sh
pip3 install -re requirements.txt
```

## 微调
#### 1. 训练数据
`data` 目录下存储了训练数据、测试数据，也可自行准备数据。数据格式如下：
```json
[
    {
        "instruction": "C++ 虚析构函数有什么作用？",
        "input": "",
        "output": "C++中，虚析构函数用于在多态场景下正确释放一个对象的资源。\n\n当一个类中存在虚函数时，该类通常也应该拥有一个虚析构函数。虚析构函数能够确保当通过基类指针删除一个派生类对象时，调用正确的析构函数，从而正确释放对象的资源。\n\n如果派生类中的析构函数不是虚函数，那么当通过基类指针删除派生类对象时，只会调用基类的析构函数，而不会调用派生类的析构函数。这可能导致派生类对象中的资源无法被正确释放，造成内存泄漏等问题。\n\n因此，使用虚析构函数可以确保在多态场景下正确释放对象资源，防止内存泄漏等问题的发生。"
    }
]
```

#### 2. Lora 微调
**（1）直接在预训练模型上监督微调**

- CUDA_VISIBLE_DEVICES=0：表示单卡训练
- --model_name_or_path：预训练模型路径
- --dataset：训练数据集
- --output_dir：微调后的模型保存路径


```sh
CUDA_VISIBLE_DEVICES=0 python src/finetune_lora.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_train \
    --dataset_dir data \
    --dataset cpp_interview_train \
    --max_source_length 256 \
    --max_target_length 3000 \
    --output_dir path_to_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16  \
    --lora_target W_pack
```

**（2）测试微调后的模型**

- --model_name_or_path：预训练模型路径
- --checkpoint_dir：微调模型路径
- --output_dir：测试结果保存路径

```sh
CUDA_VISIBLE_DEVICES=0 python src/finetune_lora.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_eval \
    --dataset_dir data \
    --dataset cpp_interview_test \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 4 \
    --max_samples 100 \
    --predict_with_generate
```

**（3）与大模型对话**

- --model_name_or_path：预训练模型路径
- --checkpoint_dir：微调模型路径

```sh
python src/cli_demo.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --checkpoint_dir path_to_lora_checkpoint \
    --max_new_tokens 2048
```

## Acknowledgement
本项目是基于 [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 项目中的 LoRA 微调部分修改而来，在此表示感谢！
