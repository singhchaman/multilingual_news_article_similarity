#!/bin/bash

# Setup virtual Env
python -m venv news_sim
source news_sim/bin/activate

# Install scraper t parse data from web
pip install semeval_8_2022_ia_downloader

# The following command scrapes the data from web, 
# Takes about 10 hours on a single machine, 
# Not required to run, We have added proccesed data
#python -m semeval_8_2022_ia_downloader.cli --links_file=semeval-2022_task8_train-data_batch.csv --dump_dir=data

# Install required modules and pretrained model
pip install ctranslate2, OpenNMT-py, tensorflow_text, sentence_transformers

# Install the pre-trained Open-NMT model for English to German translation
# We have also included tar file, so downlaoding can be skipped
# wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

# Run the translation script which preprocess and 
# translates if we have different language pair
# This also takes some time if being run on full data
# Can be skipped, we have added the processed data
python translation.py 

# Run the file to run all the models
# Default is to use stored embeddings
# Give argument as True if claculating embeddings again
python run.py False