import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import utils
import time
import argparse

import cProfile
import psutil
import time
import csv

import matplotlib.pyplot as plt
import seaborn as sns
import re
from memory_profiler import profile

def plot_memory_usage(data_str, total_loading_time, input_directory, gpu_name):
    """
    Plots memory usage and model details.

    :param data_str: String containing memory usage data.
    :param total_loading_time: Total time taken to load the model.
    :param input_directory: Directory where the model is loaded from.
    :param gpu_name: Name of the GPU used.
    """


    parsed_values = re.findall(r'(\d+,\d+|\d+)MiB', data_str)
    parsed_values = [int(val.replace(',', '')) for val in parsed_values]


    cpu_maxrss, cpu_free, gpu_used, gpu_free, gpu_total, torch_reserved, torch_allocated = parsed_values[:7]

    # Data for plotting
    categories = ['CPU Maxrss', 'CPU Free', 'GPU Used', 'GPU Free', 'GPU Total', 'Torch Reserved', 'Torch Allocated']
    values = [cpu_maxrss, cpu_free, gpu_used, gpu_free, gpu_total, torch_reserved, torch_allocated]

    # Optional: Use Seaborn for better styling
    sns.set_theme()

    # Creating the bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(categories, values, color=sns.color_palette("viridis", len(categories)))

    # Adding titles, labels, and annotations
    plt.title('Memory Usage Metrics and Model Details')
    plt.ylabel('Memory (MiB)')
    plt.xlabel('Categories')

    additional_info = f"Total Loading Time: {total_loading_time} s\nInput Directory: {input_directory}\nGPU: {gpu_name}"
    plt.annotate(additional_info, xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="#cccccc", facecolor="#ffffff"))


    plt.savefig(f"memory_usage.png")
    plt.show()

@profile
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
    meory_usage_data = f"Python Used RAM: {utils.get_mem_usage()}"
    plot_memory_usage(meory_usage_data, end_time - start_time, model_id, utils.get_gpu_name())
    print(f"Model loading time: {end_time - start_time} seconds")
    print(f"GPU: {utils.get_gpu_name()}")
    print(f"Python Used RAM: {utils.get_mem_usage()}")


def profile_function(args):
    profiler = cProfile.Profile()
    profiler.enable()

    load_model(args)

    profiler.disable()
    profiler.dump_stats('load_model_profiling.prof')

def monitor_resources():
    cpu_usage = []
    start_time = time.time()
    while time.time() - start_time < 20:  # Monitor for 10 seconds, adjust as needed
        cpu_usage.append(psutil.cpu_percent(interval=0.01))
    return cpu_usage

def save_to_csv(data, filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for i, usage in enumerate(data):
            writer.writerow([i, usage])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, help="Model directory path")
    args = parser.parse_args()

    profile_function(args)
    cpu_usage_data = monitor_resources()
    save_to_csv(cpu_usage_data, 'new_cpu_usage.csv', ['Time (s)', 'CPU Usage (%)'])
    # model = load_model(args)
    print("Model loaded successfully")


if __name__ == "__main__":
    main()