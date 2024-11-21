import torch 
from torch.utils import data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import re, os
from tqdm import tqdm 
#BrainWaves_to_get_custom_seconds_data
class BrainWaves_SP(data.Dataset):

    
    def __init__(self, kind = "train", normalize = 1):

        self.kind = kind
        df1 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data02_all\\data02.csv")
        df2 = pd.read_csv("D:\\Vikas\\lie_detection\\BrainWaves\\data\\pre-processed_raw\\data03_all\\data03.csv")

        # take only first 7000 from df1 and 2000 from df2, representing 85 percent for train and validation. 
        df1 = df1[:7000]
        df2 = df2[:2000]

        df = pd.concat((df1, df2), axis=0, )

        # Shuffle the resulting dataframe
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)  


        if kind == "train":
            df = df[:7650]
            
        elif kind == "val":
            df = df[7650:9000]

        
        # # remove this: its here for sanity
        # df = df[:100]
        df["annotations"] = df["annotations"].replace({"Present" : 1, "Absent" : 0})
        self.labels = df["annotations"].to_list()

        file_paths = df['paths']

        r_features = np.array([np.load(file_path , allow_pickle= 1) for file_path in tqdm(file_paths)], dtype= np.float64)
        


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
    

# x = BrainWaves_SP()