#!/bin/bash

# Update packages
apt update && apt upgrade -y

# Install nano editor
apt install nano -y

# Install pip package 'virtualenv'
pip install virtualenv

# Create a virtual environment named 'fint'
virtualenv fint

# Activate the virtual environment
source fint/bin/activate

# Install the 'transformers' and 'accelerate' packages
pip install transformers accelerate packaging

# Install the package in editable mode with optional dependencies
pip install -e '.[flash-attn,deepspeed]'

# Upgrade 'flash_attn'
pip install --upgrade flash_attn
