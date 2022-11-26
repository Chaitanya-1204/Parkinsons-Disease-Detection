import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import matplotlib.image as mpimg

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
    if((os.path.exists(patient_img_path))):
        print("File already exists")
    else:

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
    if((os.path.exists(healthy_img_path))):
        print("File already exists")
    else:
        img_path = os.path.join(folder_path , "healthy\V01HE02.png")
        img = cv2.imread(img_path)
        plt.figure(figsize=(5 , 5)) 
        plt.title("Healthy")
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("images/healthy.png")
        plt.close()
        print("Healthy image saved")
    






