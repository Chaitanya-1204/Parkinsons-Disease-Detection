import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import matplotlib.image as mpimg
import tensorflow as tf
import os 



def data_visualization(folder_path):

    """
    This function visualizes the data in the folder_path

    :param folder_path: it contains the path to the folder
    
    :return: Nothing but saves the image in the folder
    
    """
    os.makedirs("images" , exist_ok=True)
    
    ## for patients spiral drawings

    patient_img_path = os.path.join("images\patient.png")
    
    if not os.path.exists(patient_img_path):
        img_path = os.path.join(folder_path , "patient\V01PE02.png")
        img = mpimg.imread(img_path)
        plt.figure(figsize=(5 , 5)) 
        plt.title("Patient")
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("images/patient.png")
        plt.close()
        print("Patient image saved")

    ## for healthy persons spiral drawings
    healthy_img_path = os.path.join("images\healthy.png")

    if not os.path.exists(healthy_img_path):
        img_path = os.path.join(folder_path , "healthy\V01HE02.png")
        img = cv2.imread(img_path)
        plt.figure(figsize=(5 , 5)) 
        plt.title("Healthy")
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("images/healthy.png")
        plt.close()
        print("Healthy image saved")
    
def get_dataset(train_dataset_path , test_dataset_path):
    
    """
    
    :param 
        train_dataset_path: it contains the path to the train dataset
        test_dataset_path: it contains the path to the test dataset

    This function takes the path of dataset and make a batches of images and returns the following dataset.

    validation split is 0.1 percentage of training dataset


    
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory = train_dataset_path, 
                                                               validation_split = 0.1 , 
                                                               subset = "training",seed = 1337)
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(directory = train_dataset_path , 
                                                               validation_split = 0.1 , 
                                                               subset = "validation" , 
                                                               seed = 1337)
    testing_ds = tf.keras.preprocessing.image_dataset_from_directory(directory = test_dataset_path)

    return train_ds, validation_ds, testing_ds
                                                        








