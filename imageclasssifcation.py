import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def whatdoesmobilenetthinkofdata():
    current = os.getcwd()
    # acess all the img in a folder using keras
    for file in os.listdir(os.path.join(current,"data","PVNS")):
        file_path = os.path.join(current,"data","PVNS",file)
        # load image and resize to 224x224 for MobileNetV2
        img = image.load_img(file_path, target_size=(224, 224))
    plt.imshow(img)

    # making the image 4D for mobilenet
    resized_igm = image.img_to_array(img)
    final_image = np.expand_dims(resized_igm, axis=0)
    final_image = tf.keras.applications.mobilenet_v2.preprocess_input(final_image)

    print(final_image.shape)  # should now be (1, 224, 224, 3)

    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
    prediction = mobile.predict(final_image)
    results = imagenet_utils.decode_predictions(prediction)
    print(results)

def bycv2importimage():
    current = os.getcwd()
    # to get an image using opencv
    file_path= (os.path.join(current,"data","PVNS","Pigmented_Villonodular_Synovitis(PVNS)of_Ankle497.jpg"))
    imgg = cv2.imread(file_path)
    plt.imshow(imgg) #it will show in sgbr and not srgb

def make_csv(folder):
    current = os.getcwd()
    root = os.path.join(current, "data", folder)
    rows=[]
    for filename in os.listdir(root):
        if not filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.dcm')):
            continue
        file_path = os.path.join(root,filename)
        rows.append({"image_path":file_path,"label":folder})
    df = pd.DataFrame(rows)
    csvfilename= folder+"_path.csv"
    properpathofcsv= os.path.join(current, "data",csvfilename)
    df.to_csv(properpathofcsv,index=False)
    print(f"{folder}_path.csv created")

def train_binary_classifier(folder_a, folder_b, epochs=10, batch_size=16, random_seed=42):
    """
    folder_a, folder_b: folder names under ./data/
    returns test accuracy
    """
    current = os.getcwd()
    data_root = os.path.join(current, "data")

    # create CSVs for both folders if not exists
    make_csv(folder_a)
    make_csv(folder_b)

    # load CSVs
    df_a = pd.read_csv(os.path.join(data_root,f"{folder_a}_path.csv"))
    df_b = pd.read_csv(os.path.join(data_root,f"{folder_b}_path.csv"))

    # combine
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print("Total images:", len(df))

    # split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=random_seed, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.66, random_state=random_seed, stratify=temp_df['label'])

    # image generators
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label',
        target_size=(224,224),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )
    val_data = test_gen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label',
        target_size=(224,224),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False
    )
    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='label',
        target_size=(224,224),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False
    )

    # mobilenetv2 base
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    # custom head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # checkpoint
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')

    # fit
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    # evaluate
    test_loss, test_acc = model.evaluate(test_data)
    print("Test accuracy:", test_acc)

    return test_acc









