# EEG-Based P300 Detection (Lie detection)
This project explores advanced methodologies for EEG-based deception detection using neurophysiological signals. Below is a concise summary of key sections from the report.  

---

## 🔍 Methodology for Data Analysis  
- **Preprocessing Steps**:  
  - **Filtering**: Applied 4th-order Butterworth filter (0.5–30 Hz).  
  - **Normalization**: Ensured consistent signal scaling.  
  - **Signal Cropping**: Focused on the **250–1000 ms** window for P300 response analysis.  
- **Balancing the Dataset**:  
  - Used **SMOTE** to generate synthetic samples.  
  - Applied **EEG-specific augmentation** for better generalization.  

---

## 🤖 Model Architectures  
We evaluated several state-of-the-art deep learning models:  

| **Model**          | **Accuracy** | **Parameters (M)** | **FLOPs (MFLOPs)** |  
|---------------------|--------------|---------------------|--------------------|  
| 🏆 **CNN 1D**       | 80.61%       | 10.33               | 35.4               |  
| **ResNet**          | 77.45%       | 1.48                | 188.1              |  
| Transformer         | 60.51%       | 1.63                | 39.8               |  
| ChronoNet           | 57.14%       | 0.02                | 4.2                |  
| EfficientNet        | 59.23%       | 0.31                | 7.1                |  

---

## 📈 Results and Insights  
- **Preprocessing Impact**: Filtering and normalization improved accuracy by **~8%**.  
- **Performance Highlights**:  
  - **CNN 1D**: Best-performing model with a **ROC AUC of 0.88**.  
  - **ResNet**: Comparable accuracy with fewer parameters and computational cost.  
- **P300 Responses**: Stronger signals observed with **emotionally or personally significant probes**.  

---

## 🧠 P300 Analysis  
- **Key Finding**: Stimuli familiarity and emotional relevance influence the presence of P300 signals.  
- Neutral or unfamiliar probes often fail to elicit robust responses.  

---

## 🚀 Future Directions  
- **Multimodal Integration**: Combining EEG data with:  
  - **Skin conductance** (arousal/stress).  
  - **Heart rate** and **oxygen saturation** (physiological insights).  
- **Goal**: Achieve a richer, more accurate deception detection system.  

---

## ⚙️Steps to Run the Model
  
  - This model has been trained on a custom dataset, which cannot be shared at the moment. However, this project is designed to be easily adaptable for other datasets with minimal modifications.
  
  - Follow the steps below to use the model with your own data:

    - **Preprocess Your Data**
      - Prepare your data by creating a CSV file that includes:
        - Signal location
        - Label for each sample

    - **Data Augmentation**
      - To augment the data, run the eeg_data-augmentation.ipynb notebook. This file contains the necessary code to apply various data augmentation techniques to your dataset.

    - **Train the Model**
      - To train the model, execute the train.py script. Ensure that your dataset is properly formatted before running the script.

    - **Organize Your Data**
      - Arrange your pre-processed data in a directory structure like: ./data/pre-processed_raw/...

    Make sure your data is in the correct format for training.

    **Model Output**
    The results, along with the model checkpoints, will be saved in a directory. The output files will be named with the prefix checkpoint to indicate the saved model states.

    **Dataset Folder and Available Models**
    The dataset folder contains different configurations that support:
        - Various types of data augmentation
        Specific parameters like:
            - Normalization
            - Decimation factor
            - Start and end milliseconds for signal cropping

By following these steps and adjusting the dataset and parameters as needed, you can easily adapt the project to other datasets.
