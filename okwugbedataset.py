import os
import torch
import warnings
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from pathlib import Path
from sklearn.model_selection import train_test_split

class OkwugbeDataset(torch.utils.data.Dataset):
    """Create a Dataset for Okwugbe ASR.
    Args:
    data_type could be either 'test', 'train' or 'valid'
    """

    def __init__(self,args,
                transformation,
                target_sample_rate: int = None, 
                train_path: str =None,
                test_path: str = None,
                datatype: str = None,
                device: str = None, 
                validation_size: float =0.2):
        super(OkwugbeDataset, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.device = device
        self.validation_size = validation_size
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate
        self.original_data = None
        self.datatype = datatype.lower()
        self.args = args
        if self.train_path==None:
            raise Exception(f"`train_path` cannot be empty! You provided no path to training dataset.")
        if self.datatype =='test':
            if self.test_path==None:
                warnings.warn(f"You provided no test set. test set will be set to None and not testing evaluation will be done.")
                self.test = None
            else:
                self.test = self.load_data(self.test_path,False)

        self.train, self.validation = self.load_data(self.train_path)
        if datatype.lower() == 'train':
            self.data = self.get_data(self.train, datatype)
            self.original_data = self.train

        if datatype.lower() == 'valid':
            self.data = self.get_data(self.validation, datatype)
            self.original_data = self.validation
            # Save the validation file, if necessary
            if self.args.save_validation_data:
                self.save_pandas_to_csv(self.validation,self.args.valid_path)  

        if datatype.lower() == 'test':
            if self.test is not None: 
                self.data = self.get_data(self.test, datatype)
                self.original_data = self.test
                # Save the test data
                if self.args.save_test_data:
                    self.save_pandas_to_csv(self.test,self.args.test_path)  

            else:
                raise Exception(f"No test data was provided! Cannot request for test data")


        """datatype could be either 'test', 'train' or 'valid' """

    def load_data(self,path,split=True):
        training = pd.read_csv(path)
        
        if split:
            if self.args.valid_path is not None:
                validation = pd.read_csv(self.args.valid_path) 
            else:
                training,validation = train_test_split(training, test_size=self.validation_size,random_state=20)
            return training, validation
        else:
            return training

    def save_pandas_to_csv(self,df,filepath):
        df.to_csv(filepath,index=False)

    def get_data(self, dataset, datatype):
        data = dataset.to_numpy()
        print('{} set size: {}'.format(datatype.upper(), len(data)))
        return data

    def load_audio_item(self, d: list):
        utterance = int(d[1])
        wav_path = d[0]
        # Replace AUDIO_PATH_HERE with audio_homepath

        wav_path = wav_path.replace('AUDIO_PATH_HERE',self.args.audio_homepath)
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.to(self.device)

        waveform = self._resample_if_necessary(waveform, sample_rate)
        waveform = self.transformation(waveform,
                                    sampling_rate=self.transformation.sampling_rate, 
                                    max_length=16_000, 
                                    truncation=True,
                                    return_tensors="pt")
        return waveform.input_values, utterance

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance)``
        """
        fileid = self.data[n]
        return self.load_audio_item(fileid)

    def __len__(self) -> int:
        return len(self.data)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


