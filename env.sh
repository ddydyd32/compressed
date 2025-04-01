#!/bin/bash

pip install -q --user torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -q scikit-learn
pip install -q pyldpc
pip install -q omegaconf
pip install -q datasets
pip install -q evaluate
pip install -q transformers==4.49.0
pip install -q tf-keras
pip install -q pillow==10.1.0
pip install -q accelerate
