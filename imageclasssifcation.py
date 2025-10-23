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
    final_image = mobilenet_v2.preprocess_input(final_image)

    print(final_image.shape)  # should now be (1, 224, 224, 3)

    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
    prediction = mobile.predict(final_image)
    results = imagenet_utils.decode_predictions(prediction)
    print(results)

def bycv2importimage():
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



if __name__=="__main__":
    RANDOM_SEED = 42
    current = os.getcwd()
    data_root = os.path.join(current, "data")
    make_csv("PVNS")
    make_csv("SNN")
    pvns_df = pd.read_csv(os.path.join(data_root,"PVNS_path.csv"))
    snn_df =  pd.read_csv(os.path.join(data_root,"SNN_path.csv"))
    
    # making the df of paths 
    df = pd.concat([pvns_df, snn_df], ignore_index=True)
    # they are seperated discretly to shuffle them
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("total images are", len(df))
    print(df.head())


    # splitting the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])

    # Second split: validation vs test
    val_df, test_df = train_test_split(temp_df, test_size=0.66, random_state=42, stratify=temp_df['label'])

    #preprocessing to make it 224,224 for CNN/mobilenetv2
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
        batch_size=16,
        shuffle=True
    )
    val_data = test_gen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label',
        target_size=(224,224),
        class_mode='binary',
        batch_size=16,
        shuffle=False
    )
    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='label',
        target_size=(224,224),
        class_mode='binary',
        batch_size=16,
        shuffle=False
    )

    # load MobileNetV2 wihtout top
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    # Add custom classification head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,          # start small
        callbacks=[checkpoint]
    )
    test_loss, test_acc = model.evaluate(test_data)
    print("Test accuracy:", test_acc)

    # Optional: per-class metrics
    from sklearn.metrics import classification_report

    import numpy as np
    y_true = test_data.classes
    y_pred = (model.predict(test_data) > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))









