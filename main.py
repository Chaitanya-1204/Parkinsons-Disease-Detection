import os
from src.utilities.all_utilities import data_visualization , get_dataset
from src.models.vgg import vgg16
import tensorflow as tf

if __name__ == "__main__":

    train_dataset_path = os.path.join("F:\Parkinson_Detection\dataset\\train")
    test_dataset_path = os.path.join("F:\Parkinson_Detection\dataset\\test")
    
    data_visualization(train_dataset_path) # data visualization 

    # Getting dataset

    train_ds , test_ds , val_ds = get_dataset(train_dataset_path , test_dataset_path)

    ## Model Building 

    vgg16(train_ds , val_ds , test_ds)


    



