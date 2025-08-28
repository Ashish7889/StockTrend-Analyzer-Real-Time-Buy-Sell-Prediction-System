#!/bin/bash
set -e

# Update pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
