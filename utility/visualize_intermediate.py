"""
By: Jinseo Lee

Functions for Visualization the layers in keras models
"""

def view_model_summary(model):
    model.summary(line_length = 100)

"""
selecting layers for intermediate activation in a keras model using list of layer index
"""
def select_intermediate(model, layer_list =[]):
    layer_outputs = []
    for layer in layer_list:
        layer_outputs.append(model.layers[i].output)
    return layer_outputs

"""
selecting layers for intermediate activation in a keras model using list of layer name
"""
def select_intermediate(model, layer_list =[]):
    layer_outputs = []
    for layer_name in layer_list:
        layer_outputs.append(model.get_layer(layer_name).output)
    return layer_outputs

layer_list = ['block1_conv2','block2_conv2','block3_conv4','block4_conv4','block5_conv4']

layer_outputs = select_intermediate(model, layer_list)

from keras import models

"""
creates a new model which outputs intermediate layer activations of the selected model
"""
def process_intermediate(img, layer_outputs=[]):
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 

    intermediate_outputs = activation_model.predict(img) 
    return intermediate_outputs

intermediate_outputs = process_intermediate(np.expand_dims(img, axis=0), layer_outputs)
"""
visualizing the intermediate layer activations of the selected model
"""
def visualize_intermediate(intermediate_outputs, intermediate_ind, grid_size=(16,16)):
    # grid format
    
    feature_map_list = []
    feature_maps = intermediate_outputs[intermediate_ind]
    for i in range(feature_maps.shape[-1]):
        feature_map_list.append(feature_maps[i])
    return feature_map_list


plt.imshow(intermediate_outputs[0][0,...,0], cmap ='viridis')

def view_mult_intermediate(intermediate_outputs, save_dir = 'QuickDrZhang'):
    nrows, ncols = 3, 5  # array of sub-plots
    figsize = [11, 10]     # figure size, inches

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    layer_names = ['block 1', 'block 2', 'block 3', 'block 4', 'block 5']

    for i in range (5):
        ax[0,i].imshow(intermediate_outputs[i][0,...,0], cmap = 'viridis')
        ax[0,i].title.set_text(layer_names[i])
        ax[0,i].title.set_fontsize(15)
        ax[0,i].axis('off')
        ax[1,i].imshow(intermediate_outputs[i][0,...,31], cmap = 'viridis')
        ax[1,i].axis('off')
        ax[2,i].imshow(intermediate_outputs[i][0,...,63], cmap = 'viridis')
        ax[2,i].axis('off')


    fig.colorbar(cax = ax[0],orientation="horizontal")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    
    concat_save_dir = save_dir + '/VGG19_intermediate_feature_maps.png'
    fig.savefig(concat_save_dir)
    plt.close()
view_mult_intermediate(intermediate_outputs)
def select_filter(layer_ind, model):
    layer_outputs = model.layers[layer_ind]
    w,b = layer_outputs.get_weights()
    return w
data = np.load('Vgg16_flair_patient_11_with_GradCAM.npy')
plt.imshow(data[0], cmap = 'gray')
value = data[0]