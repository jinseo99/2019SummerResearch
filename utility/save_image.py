"""

Functions for saving generated images
"""
from matplotlib import pyplot as plt
import numpy as np

"""
Must have for ssh before plt
"""
def NoFigure():
    import matplotlib
    matplotlib.use('Agg') 
NoFigure()
plt.switch_backend('Agg')

def view_three_weighted_MRI(img, heatmap, patient = 'not specified', save_dir = 'data'):
    nrows, ncols = 1, 3  # array of sub-plots
    figsize = [11, 10]     # figure size, inches

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    weighted_title = ['flair', 't1', 't2']
    for i in range (3):
        ax[i].imshow(img[...,i], cmap = 'gray')
        ax[i].imshow(heatmap, cmap ='jet', alpha = 0.5)
        ax[i].title.set_text(weighted_title[i])
    fig.suptitle('patient number: ' + patient)
    
    print(save_dir)
    fig.savefig(save_dir +'/test.png')
    plt.close()

view_three_weighted_MRI(temp[i], CAM_1(np.expand_dims(temp[i], axis=0), -5, model))
i=100
view_three_weighted_MRI(data[i], GradCAM_3(model, np.expand_dims(data[i], axis=0), -5), save_dir = 'Figures_and_Data')

def view_three_weighted_MRI_with_original(patient_data, heatmap, patient_index, slice_index, save_dir = 'data', method_name='not_specified'):
    nrows, ncols = 2, 3  # array of sub-plots
    figsize = [11, 10]     # figure size, inches

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    MRI_type = ['flair', 't1', 't2']
    
    img = patient_data[slice_index+patient_index*135]

    for i in range (3):
        ax[0,i].imshow(img[...,i], cmap = 'gray')
        ax[0,i].title.set_text(MRI_type[i])
        ax[0,i].title.set_fontsize(15)
        ax[0,i].axis('off')
        ax[1,i].imshow(img[...,i], cmap = 'gray')
        ax[1,i].imshow(heatmap, cmap ='jet', alpha = 0.5)
        ax[1,i].axis('off')

    fig.suptitle('patient ' + str(patient_index+1) + ' | slice ' +str(slice_index), fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    concat_save_dir = save_dir + '/patient_'  + str(patient_index+1) + '_slice_' +str(slice_index)+'with_'+method_name+'.png'
    fig.savefig(concat_save_dir)
    plt.close()

data = np.load("data/38_FT1T2_data.npy")

model = loadInitialModel(3)
model.load_weights('final_Models/model_3_135_data_best_loss.h5')

i = 16

for j in [64,65,66,76,77,78]:
    img = data[j+i*135]
    view_three_weighted_MRI_with_original(data, CAM_1(np.expand_dims(img, axis=0), -5,model), i,j,save_dir = 'QuickDrZhang', method_name='CAM')
    view_three_weighted_MRI_with_original(data, GradCAM_3(model, np.expand_dims(img, axis=0), -5), i,j,save_dir = 'QuickDrZhang', method_name='GradCAM')
    view_three_weighted_MRI_with_original(data, grad_cam_plus_2(model, np.expand_dims(img, axis=0), -5), i,j,save_dir = 'QuickDrZhang', method_name='GradCAM_plusplus')

view_three_weighted_MRI_with_original(data, GradCAM_3(model, np.expand_dims(data[i], axis=0), -5), i,j,save_dir = 'Figures_and_Data')

"""
135 per patient
"""
def generate_MRI(patient_data, MRI_index = 0, patient_index = 0, save_dir = 'data'):
    
    MRI_type = ['flair','t1','t2']
    nrows, ncols = 12, 12  # array of sub-plots
    figsize = [8, 8]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    
    for i, axi in enumerate(ax.flat):
        if i >= 135:
            fig.delaxes(axi)
            continue
        img = patient_data[i+patient_index*135]
        axi.imshow(img, cmap = 'gray')
        axi.axis('off')
    
    plt.tight_layout(True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    save_dir = save_dir + '/'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '.png'
    fig.canvas.set_window_title(save_dir)

    plt.savefig(save_dir)
    plt.close()
    
data = np.load("MLBrainData/38_FT1T2_data.npy")
for i in range(3):
    print('saving: MRI_index '+str(i))
    for j in range(38):
        i = 0
        j = 0
        print('   patient index ' + str(j+1))
        generate_MRI(data[...,i], i, j, 'final_Images')

"""
use with All_Model_Summary.sh
need following functions: GradCAM_3, loadInitialModel

model_index = 3
patient_index = 1
save_dir  = 'Figures_and_Data/GC'
patient_data = data
"""

def generate_MRI_with_GradCAM(model, patient_data, model_index = 0, patient_index = 0, save_dir = 'data',numpy_dir ='data'):
    
    MRI_type = ['flair','t1','t2']
    Model_type = ["AlexNet", "ResNet50", "Vgg16", "Vgg19", "myVgg19", "myVGG19b", "myResNet50", 'ResNet18']
    last_conv_layer_index = [-14, -5, -5, -5, -8, -8, -7, -4]
    conv_index = last_conv_layer_index[model_index]
    nrows, ncols = 12, 12  # array of sub-plots
    figsize = [8, 8]     # figure size, inches
    
    heatmap_list = []
    for MRI_index in range(3):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        for i, axi in enumerate(ax.flat):
            if i >= 135:
                fig.delaxes(axi)
                continue
            img = patient_data[i+patient_index*135]
            if MRI_index == 0:
                heatmap = GradCAM_3(model, np.expand_dims(img, axis=0), conv_index)
                heatmap_list.append(heatmap)
            else:
                heatmap = heatmap_list[i]
            axi.imshow(img[...,MRI_index], cmap = 'gray')
            axi.imshow(heatmap, cmap = 'jet', alpha = 0.5)
    
            axi.axis('off')
        concat_numpy_dir = numpy_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM.npy'

        np.save(concat_numpy_dir, np.array(heatmap_list))
        
        plt.tight_layout(True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        concat_save_dir = save_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM.png'
        fig.canvas.set_window_title(concat_save_dir)

        fig.savefig(concat_save_dir)
        plt.close()

data = np.load("MLBrainData/38_FT1T2_data.npy")
print('printing numpy size: ' + str(len(data)))
#for i in range(8):
i = 4
model = loadInitialModel(i)
model.load_weights('final_Models/model_' + str(i)+'_135_data_best_loss.h5')

print('saving: Model_index '+str(i))
for j in range(38):
    print('   patient index ' + str(j+1))

    generate_MRI_with_GradCAM(model, data, i, j, 'final_Images/GC', 'final_numpy/GC')

i = 100
patient_index = 12
img = data[i+patient_index*135]
t2 = img[...,2]
plt.imshow(t2, cmap = 'gray')

print('test')


def generate_MRI_with_CAM(model, patient_data, model_index = 0, patient_index = 0, save_dir = 'data',numpy_dir ='data'):
    
    MRI_type = ['flair','t1','t2']
    Model_type = ["AlexNet", "ResNet50", "Vgg16", "Vgg19", "myVgg19", "myVGG19b", "myResNet50", 'ResNet18']
    last_conv_layer_index = [-14, -5, -5, -5, -8, -8, -7, -4]
    conv_index = last_conv_layer_index[model_index]
    nrows, ncols = 12, 12  # array of sub-plots
    figsize = [8, 8]     # figure size, inches
    
    heatmap_list = []
    for MRI_index in range(3):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        for i, axi in enumerate(ax.flat):
            if i >= 135:
                fig.delaxes(axi)
                continue
            img = patient_data[i+patient_index*135]
            if MRI_index == 0:
                heatmap = CAM_1(np.expand_dims(img, axis=0), conv_index, model)
                heatmap_list.append(heatmap)
            else:
                heatmap = heatmap_list[i]
            axi.imshow(img[...,MRI_index], cmap = 'gray')
            axi.imshow(heatmap, cmap = 'jet', alpha = 0.5)
    
            axi.axis('off')
        concat_numpy_dir = numpy_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_CAM.npy'

        np.save(concat_numpy_dir, np.array(heatmap_list))
        
        plt.tight_layout(True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        concat_save_dir = save_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_CAM.png'
        fig.canvas.set_window_title(concat_save_dir)

        fig.savefig(concat_save_dir)
        plt.close()
i = 2
model = loadInitialModel(i)
model.load_weights('final_Models/model_' + str(i)+'_135_data_best_loss.h5')
data = np.load("MLBrainData/38_FT1T2_data.npy")

print('saving: Model_index '+str(i))
for j in range(0,19):
    print('   patient index ' + str(j+1))
    generate_MRI_with_CAM(model, data, i, j, 'final_Images/CAM', 'final_numpy/CAM')

    #generate_MRI_with_GradCAM(model, data, i, j, 'final_Images/CAM', 'final_numpy/CAM')
    # to use in ARC
    
print('test')

def generate_MRI_with_GradCAM_plusplus(model, patient_data, model_index = 0, patient_index = 0, save_dir = 'data',numpy_dir ='data'):
    
    MRI_type = ['flair','t1','t2']
    Model_type = ["AlexNet", "ResNet50", "Vgg16", "Vgg19", "myVgg19", "myVGG19b", "myResNet50", 'ResNet18']
    last_conv_layer_index = [-14, -5, -5, -5, -8, -8, -7, -4]
    conv_index = last_conv_layer_index[model_index]
    nrows, ncols = 12, 12  # array of sub-plots
    figsize = [8, 8]     # figure size, inches
    
    heatmap_list = []
    for MRI_index in range(3):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        for i, axi in enumerate(ax.flat):
            if i >= 135:
                fig.delaxes(axi)
                continue
            img = patient_data[i+patient_index*135]
            if MRI_index == 0:
                heatmap = grad_cam_plus_2(model, np.expand_dims(img, axis=0), conv_index)
                heatmap_list.append(heatmap)
            else:
                heatmap = heatmap_list[i]
            axi.imshow(img[...,MRI_index], cmap = 'gray')
            axi.imshow(heatmap, cmap = 'jet', alpha = 0.5)
    
            axi.axis('off')
        concat_numpy_dir = numpy_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM_plusplus.npy'

        np.save(concat_numpy_dir, np.array(heatmap_list))
        
        plt.tight_layout(True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        concat_save_dir = save_dir + '/model_' + str(model_index) +'/'+ Model_type[model_index] +'_'+MRI_type[MRI_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM_plusplus.png'
        fig.canvas.set_window_title(concat_save_dir)

        fig.savefig(concat_save_dir)
        plt.close()

i = 2
model = loadInitialModel(i)
model.load_weights('final_Models/model_' + str(i)+'_135_data_best_loss.h5')
data = np.load("MLBrainData/38_FT1T2_data.npy")

print('saving: Model_index '+str(i))
for j in range(0,19):
    print('   patient index ' + str(j+1))
    generate_MRI_with_CAM(model, data, i, j, 'final_Images/GCPP', 'final_numpy/GCPP')


import numpy as np 
X = np.load('data/38_FT1T2_data.npy')
background = background.convert("RGBA")
overlay = overlay.convert("jet")

new_img = Image.blend(background, data[100], 0.5)
new_img.save("new.png","PNG")


def generate_data_VGG19(model, patient_data, model_index = 0, patient_index = 0, numpy_dir ='data'):
    
    Model_type = ["AlexNet", "ResNet50", "Vgg16", "Vgg19", "myVgg19", "myVGG19b", "myResNet50", 'ResNet18']
    last_conv_layer_index = [-14, -5, -5, -5, -8, -8, -7, -4]
    conv_index = last_conv_layer_index[model_index]
    
    CAM_list = []
    Grad_CAM_list = []
    Grad_CAM_plusplus_list = []
    
    for i in range(156):
        img = patient_data[i+patient_index*156]
        CAM_list.append(CAM_1(np.expand_dims(img, axis=0), conv_index, model))
        Grad_CAM_list.append(GradCAM_3(model, np.expand_dims(img, axis=0), conv_index))
        Grad_CAM_plusplus_list.append(grad_cam_plus_2(model, np.expand_dims(img, axis=0), conv_index))

    CAM_numpy_dir = numpy_dir + '/CAM/model_' + str(model_index) +'/'+ Model_type[model_index] +'_patient_'+str(patient_index+1) + '_with_CAM.npy'
    GradCAM_numpy_dir = numpy_dir + '/GC/model_' + str(model_index) +'/'+ Model_type[model_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM_plusplus.npy'
    GradCAM_plusplus_numpy_dir = numpy_dir + '/GCPP/model_' + str(model_index) +'/'+ Model_type[model_index] +'_patient_'+str(patient_index+1) + '_with_GradCAM_plusplus.npy'
    np.save(CAM_numpy_dir, np.array(CAM_list))
    np.save(GradCAM_numpy_dir, np.array(Grad_CAM_list))
    np.save(GradCAM_plusplus_numpy_dir, np.array(Grad_CAM_plusplus_list))

i = 3
X = np.load('MLBrainData/patientImagesCor.npy')
model = loadInitialModel(i)
model.load_weights('final_Models/model_' + str(i)+'_135_data_best_loss.h5')
for j in range(0,38):
    generate_data_VGG19(model, X, model_index = i, patient_index = j, numpy_dir ='final_numpy')