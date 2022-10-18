# afro-digits-speech

Project for the African Digits Recording Sprint: https://huggingface.co/spaces/chrisjay/afro-speech

The purpose of this project is to show the effectiveness of community-based crowd-sourcing dataset curation in the development of technologies for African languages.


###

- Validation data are saved [here](metrics_valid).
- The data is a csv where the first column is the audio path and the second column is the transcription.
- `train.py` is the main codebase that handles everything (from creating the datalader,training/finetuning, to pushing trained model to hub).
- `train.py` takes 3 arguments: 
    1. the file name
    2. the path to the train data (this will usually be split to training and validation set internally)
    3. [OPTIONAL] the path to the test data. If this argument is not giveb (which was our case because the data set was too small to afford a held out  test set), evaluation will only be done on the validation set.




### Reproducing the experiments

1. Clone the repo: `https://github.com/chrisemezue/afro-digits-speech.git`
2. Create a python environment and install the required packages `pip install -r requirements.txt`
3. Clone the afro digits dataset: `git clone https://huggingface.co/datasets/chrisjay/crowd-speech-africa`. 
4. After cloning , your `AUDIO_HOMEPATH` is the directory where you cloned it including the `data` folder. For example, if I cloned the dataset in the `my_personal_dir` directorym then my `AUDIO_HOMEPATH` is `my_personal_dir/crowd-speech-africa/data`. 
5. To perform one round of finetuning run the codeblock below:

```bash
python -m pdb train.py \
--experiment_directory=/home/mila/c/chris.emezue/afro-digits-speech/test \
--audio_homepath=AUDIO_HOMEPATH \
--filename=afro_ibo \
--save_model_path=/home/mila/c/chris.emezue/scratch/afr/test \
--train_path=/home/mila/c/chris.emezue/afro-digits-speech/training_data/igbo_ibo_audio_data.csv \
--valid_path=/home/mila/c/chris.emezue/afro-digits-speech/validation_data/VALID_igbo_ibo_audio_data.csv \
--num_epochs=1
```

| Please refer [here](https://github.com/chrisemezue/afro-digits-speech/blob/main/train.py#L188) for the full list of arguments and their meanings.

- Refer [`job.sh`](job.sh) for the job file to run it on cluster 