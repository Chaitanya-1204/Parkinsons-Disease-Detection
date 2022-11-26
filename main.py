import os
from src.utilities.all_utilities import data_visualization
if __name__ == "__main__":

    train_dataset_path = os.path.join("F:\Parkinson_Detection\dataset\\train")
    test_dataset_path = os.path.join("F:\Parkinson_Detection\dataset\\gotest")
    
    data_visualization(train_dataset_path)


