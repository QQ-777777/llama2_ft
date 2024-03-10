from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default="./config.json", type=str, required=False, help='path of config file')
args = parser.parse_args()

# 加载config文件
with open(args.config_path, "r") as f:
    config = json.load(f)["merge_model"]
    print("="*10, "config配置", "="*10)
    print(config)

# 设置预训练、最新微调的、merge后输出的模型路径
model_name_or_path = config["pretain_model_path"]

basename = [basename.split("/")[-1] for basename in os.listdir(config["ft_model_path"])]
basename.sort()
adapter_name_or_path = os.path.join(config["ft_model_path"], basename[-1])

from datetime import datetime
save_path = os.path.join(config["save_path"], datetime.now().strftime("%Y%m%d%s"))

# 加载分词器
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")

# 加载微调后的模型
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter ", adapter_name_or_path, " success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
