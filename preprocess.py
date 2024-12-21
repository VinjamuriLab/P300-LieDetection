import pandas as pd
import numpy as np,os

"""This code organizes the numpy files, storing their file paths and corresponding labels in a CSV file. This structured format allows easy integration with
 the dataset class for model processing."""

def process():

    df = pd.DataFrame()
    sub_paths = ["data03_all.npz", "data02_all.npz"]
    path = "./data/new_data/"
    
    dest_path = "./data/pre-processed_raw/"
    
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    for sub_path in sub_paths:
        
        sub_path_to_save = sub_path.replace("_all.npz", "_")
        internal_path = dest_path + sub_path.replace(".npz", "/")

        if not os.path.exists(internal_path): os.mkdir(internal_path)

        data = np.load(path + sub_path, allow_pickle= 1)
        
        eeg_filt = np.array(data["eeg_raw"]) # can take two values (eeg_raw, eeg_filt)
        subject_memory = np.array(data["subject_memory"])
        validity = np.array(data['validity'])
        markers = np.array(data["markers"])
        annotations = np.array(data["annotations"])
        test_id = np.array(data["annotations"]) 
        
        
        subject_memory_s = []
        validity_s = []
        markers_s = []
        annotations_s = []
        test_id_s = [] 
        
        locations = []
        indexer = 1

        for i in range(eeg_filt.shape[0]):
            
            # print(annotations[i])
            if annotations[i] not in  ["Present", "Absent", 0, 1, "0", "1"]:
                continue
            else:
                # print(internal_path, indexer, sub_path, i,f"{internal_path}/{sub_path_to_save}{indexer}.npy" )
                save_file_name = f"{internal_path}/{sub_path_to_save}{indexer}.npy"
                np.save(save_file_name, eeg_filt[i])
                

            subject_memory_s.append(subject_memory[i])
            validity_s.append(validity[i])
            markers_s.append(markers[i])
            annotations_s.append(annotations[i])
            test_id_s.append(test_id[i])
            locations.append(save_file_name)
            indexer += 1
        


        to_save = {
            "paths" : locations, 
            "annotations" : annotations_s,
            "markers" : markers_s,
            "subject_memory" : subject_memory_s,
            "validity" : validity_s,
            "test_id" : test_id_s

        }
        df = pd.DataFrame(
            to_save
        
        )
        
        df.to_csv(f"{internal_path}{sub_path[:6]}.csv")
        print("done")

process()