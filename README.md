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
3. Assume your training data path is at `/..igbo_ibo_audio_data.csv` and the name you want for the experiment is `afro_ibo`, then you run experiments the following way below:
```bash
python train.py afro_ibo /..igbo_ibo_audio_data.csv  
```


