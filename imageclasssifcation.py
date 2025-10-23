import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

current = os.getcwd()
def whatdoesmobilenetthinkofdata():
    # acess all the img in a folder using keras
    for file in os.listdir(os.path.join(current,"data","PVNS")):
        file_path = os.path.join(current,"data","PVNS",file)
        # load image and resize to 224x224 for MobileNetV2
        img = image.load_img(file_path, target_size=(224, 224))
    plt.imshow(img)

    # making the image 4D for mobilenet
    resized_igm = image.img_to_array(img)
    final_image = np.expand_dims(resized_igm, axis=0)
    final_image = mobilenet_v2.preprocess_input(final_image)

    print(final_image.shape)  # should now be (1, 224, 224, 3)

    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
    prediction = mobile.predict(final_image)
    results = imagenet_utils.decode_predictions(prediction)
    print(results)

whatdoesmobilenetthinkofdata()
def bycv2importimage():
    # to get an image using opencv
    file_path= (os.path.join(current,"data","PVNS","Pigmented_Villonodular_Synovitis(PVNS)of_Ankle497.jpg"))
    imgg = cv2.imread(file_path)
    plt.imshow(imgg) #it will show in sgbr and not srgb