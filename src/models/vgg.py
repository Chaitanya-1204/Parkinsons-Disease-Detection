import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D , Dense , Input
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping
import os
from tensorflow.keras.utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt

def vgg16(train_ds , val_ds , test_ds ):

    """
    VGG 16-layer model

    Args:
        train_ds: training dataset
        val_ds: validation dataset
        test_ds: test dataset

    """
    ## MODELS 
    base_model = tf.keras.applications.VGG16(include_top = False , input_shape = (256 , 256 , 3))
    base_model.trainable = False
    input = Input(shape = (256 , 256 , 3))
    model = base_model(input)
    model = GlobalAveragePooling2D()(model)
    output = Dense(1 , activation = "sigmoid")(model)
    model = Model(inputs = input , outputs = output)


    ## Saving MODELS PLOT 
    os.makedirs("saved_models_plot" , exist_ok = True)
    model_img_path = os.path.join("saved_models_plot/vgg16.jpeg" )
    if not os.path.exists(model_img_path):
        plot_model(model , to_file = model_img_path , show_shapes = True)
    
    ## TRAINING MODEL 
    if not os.path.exists("saved_models/vgg16.h5"):
        model.compile(optimizer = "adam" ,  loss= "binary_crossentropy" , metrics = ["accuracy"])
        history = model.fit(train_ds , validation_data = val_ds , epochs = 100 )

    os.makedirs("performance/csv_files" , exist_ok = True)
    os.makedirs("performance/plots/accuracy" , exist_ok = True)
    os.makedirs("performance/plots/loss" , exist_ok = True)

    ## Saving STATS


    if not os.path.exists("performance/csv_files/VGG16.csv"):
        pd.DataFrame(history.history).to_csv("performance/csv_files/VGG16.csv")

    ## Saving Accuracy Plot
    if not os.path.exists("performance/plots/accuracy/VGG16_accuracy.jpg"):
        accuracy_plot_path = os.path.join("performance/plots/accuracy/VGG16_accuracy.jpg")
        pd.DataFrame(history.history)[["accuracy" , "val_accuracy"]].plot()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("VGG16")
        plt.savefig(accuracy_plot_path)
        plt.close()

    ## Saving Loss Plot
    if not os.path.exists("performance/plots/loss/VGG16_loss.jpg"):
        loss_plot_path = os.path.join("performance/plots/accuracy/VGG16_loss.jpg")
        pd.DataFrame(history.history)[["loss" , "val_loss"]].plot()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("VGG16")
        plt.savefig(loss_plot_path)
        plt.close()

    print("VGG 16 Model saved and various stats and plots saved")




    