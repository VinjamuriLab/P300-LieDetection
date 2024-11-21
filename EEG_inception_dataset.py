import torch 
from torch.utils import data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import re, os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import transformers
from transformers import Transformer
from tqdm import tqdm 

class PrintTransformer(Transformer):
    def __init__(self, name=""):
        self.name = name

    def transform(self, x):
        # print(x.shape, x)
        x = np.stack(x, axis=0)
        # print(x.shape , self.name)

        return x 
    
#BrainWaves_to_get_custom_seconds_data
class EEG_inception(data.Dataset):

    
    def __init__(self, kind = "train", normalize = 1):

        self.kind = kind
        df1 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data02_all\\data02.csv")
        df2 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data03_all\\data03.csv")

       
        df = pd.concat((df1, df2), axis=0, )

        # Shuffle the resulting dataframe
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)  

        
        # use this for normalizing the data. 
        sampling_rate = 1000
        # decimation_factor = 1
        # final_rate = sampling_rate // decimation_factor
        epoch_duration = 1.8 # seconds


        eeg_pipe = make_pipeline(
            # transformers.Decimator(decimation_factor),
            # PrintTransformer(name='de'),
            transformers.ButterFilter(sampling_rate , 4, 0.5, 30),
            PrintTransformer(name='bu'),
            transformers.ChannellwiseScaler(StandardScaler())
            
        )

        df["annotations"] = df["annotations"].replace({"Present" : 1, "Absent" : 0})
# markers_pipe = transformers.MarkersTransformer(labels_mapping, decimation_factor)
    
        if normalize == True or kind == "train":
            temp = df[:7650]
            
            # Separate positive and negative samples
            # pos_samples = df[df["annotations"] == 1]
            # neg_samples = df[df["annotations"] == 0]

            # # Ensure both classes have 2300 samples
            # pos_samples = pos_samples[:2300]
            # neg_samples = neg_samples[:2300]

            # # Combine the two classes back together
            # df = pd.concat([pos_samples, neg_samples], axis=0)

            # ones = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\augmented_data\\ones\\ones.csv")
            # zeroes = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\augmented_data\\zeroes\\zeroes.csv")
            # df = pd.concat((df, ones, zeroes), axis=0).reset_index(drop = True)

            if kind == "train":
                df = temp[:]
        if kind == "val":
            df = df[7650:9000]

            
            # Separate positive and negative samples
            pos_samples = df[df["annotations"] == 1]
            neg_samples = df[df["annotations"] == 0]

            # Ensure both classes have 2300 samples
            print(min(len(pos_samples), len(neg_samples)))
            pos_samples = pos_samples[:min(len(pos_samples), len(neg_samples))]
            neg_samples = neg_samples[:min(len(pos_samples), len(neg_samples))]
            
            print("here", len(df))
            # # Combine the two classes back together
            df = pd.concat([pos_samples, neg_samples], axis=0)
        

        # # remove this: its here for sanity
        # df = df[:100]
        
        self.labels = df["annotations"].to_list()

        file_paths = df['paths']

        # r_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)
        r_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)
        print(r_features.shape, "shap[e]")
        r_features = r_features[:,:,250:1000]
        if normalize == True:
            for eegs in r_features:
                eeg_pipe.fit(eegs)
        
            for indx, eegs in enumerate(r_features):
                r_features[indx] = eeg_pipe.transform(eegs)

        print(r_features.shape, "in here dataset")
        self.features = r_features
            
        self.indices = list(range(len(self.features)))
    

        print(kind, "kind",len(df), len(self.labels), len(self.features))
        # print(self.indices)
        print("main_job done")

    def __getitem__(self, indx):
     
        data = torch.tensor(self.features[indx]).float()  # Use float32 for tensor
        
        label = self.labels[indx]

        return data, label
    
    def __len__(self):
        return len(self.indices)
    

# e = EEG_inception(kind = "val" , normalize= 1)
# print(e[1])
# pos = neg = 0
# for i in e:
#     if i[1] == 1:
#         pos+=1
#     else: neg +=1 
# print(pos, neg)