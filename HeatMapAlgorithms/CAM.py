"""
By: Jinseo Lee

Class Activation Map for Models with GAP (Global Average Pooling) layer
"""
from keras.models import Model   
import scipy
import numpy as np
# from fresh/practiceImgNet
def CAM_1(img, last_conv_ind, model, img_size = (256,256)): 
    last_layer_weights = model.layers[-1].get_weights()[0]
    last_conv_layer = model.layers[last_conv_ind]
    CAM_Model = Model(inputs=model.input, outputs=(last_conv_layer.output, model.output)) 
    last_conv_output, pred_ind = CAM_Model.predict(img)
    last_conv_output = np.squeeze(last_conv_output) 
    pred = np.argmax(pred_ind)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (img_size[0]/last_conv_layer.output_shape[1], img_size[1]/last_conv_layer.output_shape[2], 1), order=1)
    pred_weights = last_layer_weights[:, pred]
    heatmap = np.dot(mat_for_mult.reshape((img_size[0]*img_size[1], last_conv_layer.output_shape[-1])), pred_weights).reshape(img_size[0],img_size[1])
    return heatmap

# probably grad_cam
def CAM_2(img, last_conv_ind, model, img_size = (256,256)): 
    preds = model.predict(x) 
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.layers[last_conv_ind] # change
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    #pooled_grads_value, conv_layer_output_value = iterate([x])
    
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(last_conv_layer.shape[-1]): # change
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    return final_output

"""
# Sample test
# Using CAM_1 on patient 7 and 11
    
i = 3+19
y = np.load('data/labels_categorical.npy')


heatmap = CAM_1(np.expand_dims(temp[i], axis=0), -5, model)
plt.figure()
plt.imshow(temp[i,...,0], cmap = 'gray')
plt.imshow(heatmap, cmap= 'jet', alpha=0.5)
i = 79

for i in [64,65,66]:
    i=i+12
    patient_index = 16
    img = data[i+patient_index*135]
    pred=model.predict(np.expand_dims(img,axis=0))
    pred = np.argmax(pred)
    ans = y[i+patient_index*135]
    if pred == ans:
        print('correct')
    else:
        print('false')
    

heatmap = CAM_1(np.expand_dims(img,axis=0), -5, model)
"""