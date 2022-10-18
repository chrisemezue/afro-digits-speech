#!/bin/bash

###load environments
module load python/3
source /home/mila/c/chris.emezue/scratch/okwugbe-afr/bin/activate


python -m pdb train.py \
--experiment_directory=/home/mila/c/chris.emezue/afro-digits-speech/test \
--audio_homepath=/home/mila/c/chris.emezue/scratch/afr/data \
--filename=afro_ibo_test_discard \
--save_model_path=/home/mila/c/chris.emezue/scratch/afr/test \
--multilingual_model_path=/home/mila/c/chris.emezue/scratch/afr/afrospeech-wav2vec-all-6.pth \
--train_path=/home/mila/c/chris.emezue/afro-digits-speech/training_data/igbo_ibo_audio_data.csv \
--valid_path=/home/mila/c/chris.emezue/afro-digits-speech/validation_data/VALID_igbo_ibo_audio_data.csv \
--num_epochs=1