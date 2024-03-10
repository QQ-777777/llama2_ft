from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaTokenizer
from peft import LoraConfig
from trl import SFTTrainer
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default="./config.json", type=str, required=False, help='path of config file')
args = parser.parse_args()

# 加载config文件
with open(args.config_path, "r") as f:
    config = json.load(f)["finetune"]
    print("="*10, "config配置", "="*10)
    print(config)

# 设置gpu
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]

# 加载FT数据集
dataset = load_dataset("json",data_files=config["ft_data_path"],split="train")

# 加载预训练模型
base_model_name =config["pretain_model_path"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,#在4bit上，进行量化
    bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
    bnb_4bit_quant_type="nf4",#NF4（normalized float）或纯FP4量化 博客说推荐NF4
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,#本地模型
    quantization_config=bnb_config,#上面本地模型的配置
    device_map="auto",#使用GPU的编号
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# 加载分词器
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 设置输出路径
from datetime import datetime
output_dir = os.path.join(config["base_output_dir"], datetime.now().strftime("%Y%m%d%s"))

# 设置训练超参数
training_args = TrainingArguments(
    # report_to="wandb",
    output_dir=output_dir, #训练后输出目录
    per_device_train_batch_size=config["per_device_train_batch_size"], #每个GPU的批处理数据量
    gradient_accumulation_steps=config["gradient_accumulation_steps"], #在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
    learning_rate=config["learning_rate"], #超参、初始学习率。太大模型不稳定，太小则模型不能收敛
    logging_steps=config["logging_steps"], #两个日志记录之间的更新步骤数
    max_steps=config["max_steps"] #要执行的训练步骤总数
)

# 设置SFT
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    tokenizer=tokenizer,
    args=training_args,
)

# 开始训练
trainer.train()

# 保存模型
trainer.model.save_pretrained(output_dir)