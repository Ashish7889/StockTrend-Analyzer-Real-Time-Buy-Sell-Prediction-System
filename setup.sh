#!/bin/bash
set -e

# Update pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
