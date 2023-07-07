
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
pip3 install -r requirements.txt
```

##### 4. 硬件要求
- 内存：建议 48GB 以上

- 显存：


## 微调
#### 1. 准备数据
`data` 目录下存储了训练数据、测试数据，也可自行准备数据。数据格式如下：
```json
[
    {
        "instruction": "C++ 虚析构函数有什么作用？",
        "input": "",
        "output": "C++中，虚析构函数用于在多态场景下正确释放一个对象的资源。\n\n当一个类中存在虚函数时，该类通常也应该拥有一个虚析构函数。虚析构函数能够确保当通过基类指针删除一个派生类对象时，调用正确的析构函数，从而正确释放对象的资源。\n\n如果派生类中的析构函数不是虚函数，那么当通过基类指针删除派生类对象时，只会调用基类的析构函数，而不会调用派生类的析构函数。这可能导致派生类对象中的资源无法被正确释放，造成内存泄漏等问题。\n\n因此，使用虚析构函数可以确保在多态场景下正确释放对象资源，防止内存泄漏等问题的发生。"
    },
     {
        "instruction": "C++ 虚函数与普通成员函数的区别是什么？",
        "input": "",
        "output": "C++中的虚函数和普通成员函数之间有几个主要的区别：\n\n1. 声明方式：虚函数通过在基类中使用关键字`virtual`进行声明，而普通成员函数没有这个关键字。\n\n2. 动态绑定：使用了虚函数的类在运行时实现了动态绑定（也称为后期绑定或运行时多态），这意味着调用虚函数的时候，实际调用的具体实现将根据对象的类型决定。而普通成员函数在编译时通过静态绑定（也称为早期绑定或编译时多态）确定调用的具体实现。\n\n3. 表现形式：虚函数通常在基类中声明，并在派生类中重写。这使得我们可以使用基类的指针或引用调用派生类对象的虚函数，以实现多态性。普通成员函数没有这种特性。\n\n4. 动态绑定的开销：虚函数的动态绑定带来了额外的开销，涉及到虚函数指针和虚函数表（vtable）的查找。而普通成员函数的调用没有这样的开销。\n\n总而言之，虚函数的主要特点是实现动态绑定，可以在运行时根据对象的类型调用不同的实现，而普通成员函数没有这个能力。虚函数的动态绑定需要额外的开销，因此在性能要求高的场景下可能不适合使用。"
    },
]
```

#### 2. Lora 微调

##### 在微调模型上继续微调

微调过的模型放置在 `baichuan_lora_checkpoint` 文件夹中，可在该模型上继续 `Lora` 微调。 

- CUDA_VISIBLE_DEVICES=0：表示单卡训练
- --model_name_or_path：预训练模型路径
- --dataset：训练数据集
- --checkpoint_dir：微调模型路径
- --output_dir：微调后的模型保存路径

```sh
CUDA_VISIBLE_DEVICES=0 python src/finetune_lora.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_train \
    --dataset_dir data \
    --dataset cpp_interview_train \
    --max_source_length 256 \
    --max_target_length 3000 \
    --checkpoint_dir baichuan_lora_checkpoint \
    --output_dir baichuan_lora_checkpoint2 \
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

##### 直接在预训练模型上微调
> `此步骤与上一步骤，可二选一`。

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
    --output_dir baichuan_lora_checkpoint \
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

##### 测试微调后的模型

- --model_name_or_path：预训练模型路径
- --checkpoint_dir：微调模型路径
- --output_dir：测试结果保存路径

```sh
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
    
```

##### 与大模型对话

- --model_name_or_path：预训练模型路径
- --checkpoint_dir：微调模型路径

```sh
python src/cli_demo.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --checkpoint_dir path_to_lora_checkpoint \
    --max_new_tokens 2048
```

##### 对话展示

## Acknowledgement
本项目是基于 [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 项目中的 LoRA 微调部分修改而来，在此表示感谢！
