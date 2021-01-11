"""
By: Jinseo Lee

Upgraded version of Gradient based Class Activation Map methods

Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks. 2018 IEEE Winter Conference on Applications of Computer Vision (WACV). doi: 10.1109/wacv.2018.00097

better explanation: https://arxiv.org/pdf/1710.11063.pdf

"""


import numpy as np
import keras.backend as K
from scipy.ndimage.interpolation import zoom
import cv2

def grad_cam_plus_1(input_model, img, layer_name,H=256,W=256):
    print('here')
    cls = np.argmax(input_model.predict(img))
    y_c = input_model.output[0, cls]

    conv_output = input_model.get_layer(layer_name).output
    #conv_output = target_conv_layer, ex. mixed10 1,5,5,2048
    grads = K.gradients(y_c, conv_output)[0]
    print('here')
    
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
    print('here')

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom
    print('here')

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0) 
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2) # 20
    print('here')

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,H/cam.shape[0])
    cam = cam / np.max(cam) # scale 0 to 1.0    
    cam = cv2.resize(cam, (H,W))

    return cam



def grad_cam_plus_2(input_model, img, layer_index,H=256,W=256):
    cls = np.argmax(input_model.predict(img))
    y_c = input_model.output[0, cls]

    conv_output = input_model.layers[layer_index].output
    #conv_output = target_conv_layer, ex. mixed10 1,5,5,2048
    grads = K.gradients(y_c, conv_output)[0]
    
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0) 
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2) # 20

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,H/cam.shape[0])
    cam = cam / np.max(cam) # scale 0 to 1.0    
    cam = cv2.resize(cam, (H,W))

    return cam

"""
# sample test

heatmap = grad_cam_plus_2(model, np.expand_dims(altered, axis =0),'block5_conv4')
plt.imshow(altered[...,0], cmap = 'gray')
plt.imshow(heatmap, cmap= 'jet', alpha=0.5)
"""