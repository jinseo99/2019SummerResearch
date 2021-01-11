"""
By: Jinseo Lee

Gradient based Class Activation Map methods
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. 2017 IEEE International Conference on Computer Vision (ICCV). doi: 10.1109/iccv.2017.74
"""
from keras import backend as K
import numpy as np
from scipy.ndimage.interpolation import zoom
import cv2

def GradCAM_1(model, input_file, layer_ind):
    preprocessed_input = input_file
    
    predictions = model.predict(preprocessed_input)
    cls = np.argmax(predictions)
    y_c = model.output[0, cls]
    conv_output = model.layers[layer_ind].output # change this li =-5
    grads = K.gradients(y_c, conv_output)[0]
    
    gradient_function = K.function([model.input], [conv_output, grads])
    
    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0,...], grads_val[0,...]
    
    weights = np.mean(grads_val, axis=(0, 1, 2))
    grad_cam = np.ones(output.shape[:-1], dtype=K.floatx())
    for i, w in enumerate(np.transpose(weights)):
        grad_cam += w * output[..., i]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam / np.max(grad_cam)
    activation_map = np.zeros_like(input_file[0,:,:,:,0])
    
    zoom_factor = [model.input_shape[1]/model.layers[layer_ind].output_shape[1],
                   model.input_shape[2]/model.layers[layer_ind].output_shape[2],
                   model.input_shape[3]/model.layers[layer_ind].output_shape[3]]#[i / (j * 1.0) for i, j in iter(zip(input_file.shape, grad_cam.shape))]
    activation_map = zoom(grad_cam, zoom_factor)

    activation_map = (activation_map / np.max(activation_map))
    return activation_map


# part of GradCAM_2
def loss_calculation(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot((category_index), nb_classes))

def loss_calculation_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def loss_function(x):
    loss_calculation(x,1,nb_classes)
    
def prepareGradCAM(input_model, conv_layer_index, nb_classes):
    model = input_model
    #  because non-manufacturability is 1
    explanation_catagory = 1
    loss_function = lambda x: loss_calculation(x, 1, nb_classes)
    model.add(Lambda(loss_function, output_shape=loss_calculation_shape))
    #  use the loss from the layer before softmax. As best practices
    loss = K.sum(model.layers[-1].output)
    # last fully Convolutional layer to use for computing GradCAM
    conv_output = model.layers[-6].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])
    
    return gradient_function

def GradCAM_2():
    output, grads_val = gradient_function([newdata])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    weights = np.mean(grads_val, axis = (0, 1,2))
    cam = np.ones(output.shape[0 : 3], dtype = np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, :,i]
    
    cam = cv2.resize(cam, (10, 256, 256))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    
def GradCAM_3(model, img, conv_index):
    
    predictions = model.predict(img)
    cls = np.argmax(predictions)
    y_c = model.output[0, cls]
    conv_output = model.layers[conv_index].output 

    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([model.input], [conv_output, grads])
    
    output, grads_val = gradient_function([img])
    output, grads_val = output[0], grads_val[0]
    
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    
    cam = cv2.resize(cam, (256, 256), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
        
    return cam

"""
# Sample test

heatmap = GradCAM_3(model, np.expand_dims(altered, axis =0), -5)
plt.imshow(altered[...,0], cmap = 'gray')
plt.imshow(heatmap, cmap= 'jet', alpha=0.5)

"""