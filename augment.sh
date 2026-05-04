#!/bin/bash

source venv/bin/activate

python3 src/augmentation/image_augment.py
python3 src/augmentation/audio_augment.py
