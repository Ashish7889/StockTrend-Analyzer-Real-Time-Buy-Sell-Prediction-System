#!/bin/bash
set -e

# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install base dependencies first
pip install numpy==1.24.3 pandas==1.5.3

# Install the rest of the requirements
pip install -r requirements.txt --no-deps

# Install with dependencies
pip install -r requirements.txt
pip install -e .
