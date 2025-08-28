#!/bin/bash
set -e

# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install base packages first
pip install numpy pandas

# Install the rest
pip install -r requirements.txt
