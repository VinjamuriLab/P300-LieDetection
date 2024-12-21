import torch 
from torch.utils import data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import re, os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import transformers_
from transformers_ import Transformer
from tqdm import tqdm 

# this is used to 
class Reshape(Transformer):
    """This class reshapes the output of the pipelibe object"""
    def __init__(self, name=""):
        self.name = name

    def transform(self, x):
        x = np.stack(x, axis=0)
        return x 


#BrainWaves_to_get_custom_seconds_data
class EEG_inception(data.Dataset):
 
    """
   This `Dataset` class is designed to output data based on specific variations provided. The parameters include:

        - kind: Specifies the dataset type, e.g., `train` or `val`.
        - normalize: Indicates whether data normalization is applied (`True` or `False`).
        - balancing: Defines the balancing strategy, which can be `equal_samples`, `smote`, `inception`, or `default`.
        - start_msecond: Specifies the starting point (in milliseconds) for data processing.
        - end_msecond: Specifies the endpoint (in milliseconds) for data processing.

        ### Hyperparameters:
        - balancing
        - normalize
        - decimation_factor
        - start_msecond
        - end_msecond

        *Note*: Modifying these hyperparameters requires corresponding changes to the model architecture. """
    
    def __init__(self, kind = "train", normalize = 1, balancing = "default"):

        self.kind = kind
        self.balancing = balancing

        start_msecond = 250
        end_msecond = 1000

        df1 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data02_all\\data02.csv")
        df2 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data03_all\\data03.csv")
        df = pd.concat((df1, df2), axis=0, )
        # Shuffle the resulting dataframe
        
        
        # use this for normalizing the data. 
        sampling_rate = 1000
        decimation_factor = 1 # meaning, the resultant signal is equal to 1800/decimation_factor

        # this pipeline is to filter, decimate, and normalize the data
        eeg_pipe = make_pipeline(
            transformers_.Decimator(decimation_factor),
            Reshape(name='de'),
            transformers_.ButterFilter(sampling_rate , 4, 0.5, 30),
            Reshape(name='bu'),
            transformers_.ChannellwiseScaler(StandardScaler())
            
        )

        df["annotations"] = df["annotations"].replace({"Present" : 1, "Absent" : 0})
    

        if normalize == True or kind == "train":
            original_data = df[:7650].copy()
            balanced_data = self.balance_data(original_data.copy())
            
        if kind == "val":
            df = df[7650:9000]

            # balancing can only be equal_samples, when kind is val
            self.balancing = "equal_samples"
            balanced_data = self.balance_data(df.copy())
        
        
        if self.balancing == "smote":
            
            self.labels = original_data["annotations"].to_list()
            file_paths = original_data['paths']

            # smote the dataset
            r_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)

            r_features = r_features[:,:,start_msecond: end_msecond]
            r_features = r_features.reshape(-1, 8 * (end_msecond - start_msecond))
            print("before", r_features.shape)
            smote = SMOTE(random_state=42)
            r_features, self.labels = smote.fit_resample(r_features, self.labels)
            r_features = r_features.reshape(-1, 8, end_msecond - start_msecond)
        
        else:
            self.labels = balanced_data["annotations"].to_list()
            file_paths = balanced_data['paths']

            r_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)
            r_features = r_features[:,:,start_msecond:end_msecond]


        if normalize == True:
            
            file_paths = original_data['paths']

            # smote the dataset
            orig_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)

            for eegs in orig_features:
                eeg_pipe.fit(eegs)
            
            for indx, eegs in enumerate(r_features):
                r_features[indx] = eeg_pipe.transform(eegs)
            
        self.features = r_features
        self.indices = list(range(len(self.features)))
    

    def __getitem__(self, indx):
     
        data = torch.tensor(self.features[indx]).float()  # Use float32 for tensor
        
        label = self.labels[indx]

        return data, label
    
    def __len__(self):
        return len(self.indices)


    def balance_data(self, df):
        if self.balancing == "equal_samples":
            
            # Separate positive and negative samples
            pos_samples = df[df["annotations"] == 1][:]
            neg_samples = df[df["annotations"] == 0][:]

            # Separate positive and negative samples
            pos_samples = pos_samples[:min(len(pos_samples), len(neg_samples))]
            neg_samples = neg_samples[:min(len(pos_samples), len(neg_samples))]
            
            # # Combine the two classes back together
            df = pd.concat([pos_samples, neg_samples], axis=0)

        elif self.balancing == "inception":
            ones = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\augmented_data\\ones\\ones.csv")
            zeroes = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\augmented_data\\zeroes\\zeroes.csv")
            df = pd.concat((df, ones, zeroes), axis=0).reset_index(drop = True)
        
        elif self.balancing == "smote":
            pass
        
        elif self.balancing == "default":
            return df
        
        else:
            raise "unknown balancing used, change balancing to smote, inception, None, or equal_samples"

        df = df.sample(frac=1, random_state=1).reset_index(drop=True)  
        return df

