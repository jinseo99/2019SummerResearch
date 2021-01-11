"""
By: Jinseo Lee

Various image preprocessing functions for deep learning in keras
"""

import numpy as np
from keras.preprocessing import image

"""
scale pixel value to fit keras model
"""
def preprocess_image(img_path = 'test/2.jpg', img_size = (300,300)):
    
    img = image.load_img(img_path, target_size=img_size)
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    return img_tensor

"""
Preprocesses different image transformation for 'fit_generator' in keras
Manually change parameters appropriately
"""
def preprocess_flow(transformation, img_dir = 'shape', img_size):
    
    if transformation == True:
        # https://keras.io/preprocessing/image/ for more preprocessing parameters
        datagen = image.ImageDataGenerator(
                rescale=1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip=True)
    else:
        # no transformation on Validation and Test set image data
        datagen = image.ImageDataGenerator(rescale=1./255)

    
    # batch size of 16 ~ 32 is good any higher will use up too much memory
    img_set = train_datagen.flow_from_directory(img_dir,
                                                target_size=img_size,
                                                batch_size=16,
                                                class_mode='categorical')
    return img_set

