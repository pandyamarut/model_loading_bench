import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import utils
import time
import argparse
import csv


import matplotlib.pyplot as plt
import seaborn as sns
import re
from memory_profiler import profile

# def plot_memory_usage(data_str, total_loading_time, input_directory, gpu_name):
#     """
#     Plots memory usage and model details.

#     :param data_str: String containing memory usage data.
#     :param total_loading_time: Total time taken to load the model.
#     :param input_directory: Directory where the model is loaded from.
#     :param gpu_name: Name of the GPU used.
#     """

#     # Parsing the data using regular expressions
#     parsed_values = re.findall(r'(\d+,\d+|\d+)MiB', data_str)
#     parsed_values = [int(val.replace(',', '')) for val in parsed_values]

#     # Assigning values to categories
#     cpu_maxrss, cpu_free, gpu_used, gpu_free, gpu_total, torch_reserved, torch_allocated = parsed_values[:7]

#     # Data for plotting
#     categories = ['CPU Maxrss', 'CPU Free', 'GPU Used', 'GPU Free', 'GPU Total', 'Torch Reserved', 'Torch Allocated']
#     values = [cpu_maxrss, cpu_free, gpu_used, gpu_free, gpu_total, torch_reserved, torch_allocated]

#     # Optional: Use Seaborn for better styling
#     sns.set_theme()

#     # Creating the bar chart
#     plt.figure(figsize=(14, 8))
#     plt.bar(categories, values, color=sns.color_palette("viridis", len(categories)))

#     # Adding titles, labels, and annotations
#     plt.title('Memory Usage Metrics and Model Details')
#     plt.ylabel('Memory (MiB)')
#     plt.xlabel('Categories')

#     # Annotations in the top right corner
#     additional_info = f"Total Loading Time: {total_loading_time} s\nInput Directory: {input_directory}\nGPU: {gpu_name}"
#     plt.annotate(additional_info, xy=(0.95, 0.95), xycoords='axes fraction',
#                  ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="#cccccc", facecolor="#ffffff"))

#     # Save and show the plot
#     plt.savefig(f"memory_usage.png")
#     plt.show()
def plot_memory_usage(data_str, total_loading_time, input_directory, gpu_name, csv_filename):
    """
    Plots memory usage and model details and saves them to a CSV file.

    :param data_str: String containing memory usage data.
    :param total_loading_time: Total time taken to load the model.
    :param input_directory: Directory where the model is loaded from.
    :param gpu_name: Name of the GPU used.
    :param csv_filename: Filename to save the CSV data.
    """

    # Parsing the data using regular expressions
    parsed_values = re.findall(r'(\d+,\d+|\d+)MiB', data_str)
    parsed_values = [int(val.replace(',', '')) for val in parsed_values]

    # Assigning values to categories
    categories = ['CPU Maxrss', 'CPU Free', 'GPU Used', 'GPU Free', 'GPU Total', 'Torch Reserved', 'Torch Allocated']
    values = parsed_values[:7]

    # Saving data to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Category', 'Memory (MiB)', 'Total Loading Time (s)', 'Input Directory', 'GPU Name'])
        for category, value in zip(categories, values):
            csvwriter.writerow([category, value, total_loading_time, input_directory, gpu_name])

    # Optional: Use Seaborn for better styling
    sns.set_theme()

    # Creating the bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(categories, values, color=sns.color_palette("viridis", len(categories)))

    # Adding titles, labels, and annotations
    plt.title('Memory Usage Metrics and Model Details')
    plt.ylabel('Memory (MiB)')
    plt.xlabel('Categories')

    # Annotations in the top right corner
    additional_info = f"Total Loading Time: {total_loading_time} s\nInput Directory: {input_directory}\nGPU: {gpu_name}"
    plt.annotate(additional_info, xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="#cccccc", facecolor="#ffffff"))

    # Save and show the plot
    plt.savefig("memory_usage.png")
    plt.show()

@profile
def load_model(args, csv_filename):
    model_name = args.model_directory
    before_mem = utils.get_mem_usage()
    start_time = time.time()
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,device_map="auto",
        use_safetensors=True
    )
    end_time = time.time()
    after_mem = utils.get_mem_usage()
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([model_name, before_mem, after_mem, total_bytes_str, duration])
    # plot_memory_usage(meory_usage_data, end_time - start_time, model_id, utils.get_gpu_name())
    print(f"Model loading time: {end_time - start_time} seconds")
    print(f"GPU: {utils.get_gpu_name()}")
    print(f"Python Used RAM: {utils.get_mem_usage()}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, help="Model directory path")
    args = parser.parse_args()
    csv_filename = 'tensor1.csv'

    # Initialize CSV file with headers for memory usage logging
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model Name', 'Memory Before (MiB)', 'Memory After (MiB)', 'Total Bytes', 'Loading Duration (s)'])
    model = load_model(args, csv_filename)
    
    print("Model loaded successfully")


if __name__ == "__main__":
    main()