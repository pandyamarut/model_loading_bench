# Model Loading

Output : 

```
Model loading time: 6.950401306152344 seconds

GPU: NVIDIA RTX A6000
Python Used RAM: CPU: (maxrss: 11,724MiB F: 6,454MiB) GPU: (U: 45,756MiB F: 2,920MiB T: 48,676MiB) TORCH: (R: 45,488MiB/45,488MiB, A: 45,482MiB/45,482MiB)

```

To run classic loading:

```
python src/hf_loader.py --model_directory /runpod-volume/model-7b 
```

To understand memory_usage used memory_profiler for python: 
```
example: mprof run hf_loader.py --model_directory meta-llama/Llama-2-13b-chat-hf
for ploting: mprof plot -o memory_usage_{model_name}.png
```