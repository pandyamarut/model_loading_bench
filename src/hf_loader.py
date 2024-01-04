import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import utils
import time
import argparse

def load_model(args):
    model_id = args.model_directory
    start_time = time.time()
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,device_map="auto",
        use_safetensors=True
    )
    end_time = time.time()
    print(f"Model loading time: {end_time - start_time} seconds")
    print(f"GPU: {utils.get_gpu_name()}")
    print(f"Python Used RAM: {utils.get_mem_usage()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, help="Model directory path")
    args = parser.parse_args()
    model = load_model(args)
    print("Model loaded successfully")


if __name__ == "__main__":
    main()