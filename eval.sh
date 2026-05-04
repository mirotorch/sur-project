#!/bin/bash

source venv/bin/activate
mkdir -p cache

python3 src/image_svm.py --pca-components 0.95 --mode train
python3 src/image_svm.py --pca-components 0.95 --mode eval 

python3 src/audio_cnn.py --mode train --cv-strategy loso
python3 src/audio_cnn.py --mode eval --cv-strategy loso

python3 src/fusion.py --mode train --cv-strategy loso --method combined --pca-components 0.95
python3 src/fusion.py --mode eval --cv-strategy loso --method combined --pca-components 0.95
