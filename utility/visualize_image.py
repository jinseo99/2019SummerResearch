"""
By: Jinseo Lee

Various methods for displaying data and figures
Code run in Ananaconda Spyder
"""

import numpy as np
import cv2

"""
combines multiple images into one retangular box image
"""
def mult_image_to_one(img_list, img_size = (28,28), n_rows, n_cols):
    display_grid = np.zeros((img_size[0] * n_rows, n_cols * size,3))
    for row in range(n_rows):
        for col in range(n_cols):
            img = img_list[col+row*n_cols]
            img = np.clip(img, 0, 255).astype('uint8')
            display_grid[row * img_size[0] : (row + 1) * img_size[0], col * img_size[1] : (col + 1) * img_size[1]] = img
    scale = 1. / img_size[0]
    display_grid = np.clip(display_grid, 0, 255).astype('uint8')
    
    # optional setting for plt.figure
    #plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    #plt.grid(False)
    return display_grid

def mult_image_one_fig(nrows = 1, ncols = 2, figsize=(15,15)):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, ax

def view_three_weighted_MRI(img, heatmap, patient = 'not specified'):
    fig, ax = mult_image_one_fig(1,3)
    weighted_title = ['flair', 't1', 't2']
    for i in range (3):
        ax[i].imshow(img[...,i], cmap = 'gray')
        ax[i].imshow(heatmap, cmap ='jet', alpha = 0.5)
        ax[i].title.set_text(weighted_title[i])
        ax[i].title.set_fontsize(15)
    fig.suptitle('patient number: ' + patient, fontsize=20)

view_three_weighted_MRI(temp[i], CAM_1(np.expand_dims(temp[i], axis=0), -5, model))

def view_three_weighted_MRI_with_original(img, heatmap, patient = 'not specified'):
    fig, ax = mult_image_one_fig(2,3, (11,10))
    weighted_title = ['flair', 't1', 't2']
    for i in range (3):
        ax[0,i].imshow(img[...,i], cmap = 'gray')
        ax[0,i].title.set_text(weighted_title[i])
        ax[0,i].title.set_fontsize(15)
        ax[0,i].axis('off')
        ax[1,i].imshow(img[...,i], cmap = 'gray')
        ax[1,i].imshow(heatmap, cmap ='jet', alpha = 0.5)
        ax[1,i].axis('off')

    fig.suptitle('patient number: ' + patient, fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)



i = 0
view_three_weighted_MRI_with_original(temp[i], CAM_1(np.expand_dims(temp[i], axis=0), -5, model))
view_three_weighted_MRI_with_original(temp[i+19], CAM_1(np.expand_dims(temp[i+19], axis=0), -5, model))

view_three_weighted_MRI_with_original(temp[i], GradCAM_3(model, np.expand_dims(temp[i], axis=0), -5))
view_three_weighted_MRI_with_original(temp[i+19], GradCAM_3(model, np.expand_dims(temp[i+19], axis=0), -5))

class interactive_view:
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    def multi_slice_viewer(volume, si):
        remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = si
        ax.imshow(volume[ax.index,:,:,0], cmap = 'gray', alpha=0.5)
    
        ax.set_title('slice index: ' + str(ax.index))
    
        fig.canvas.mpl_connect('key_press_event', process_key)
    
    
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()
    
    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index,:,:,0])
        ax.set_title('slice index: ' + str(ax.index))
    
    
    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index,:,:,0])
        ax.set_title('slice index: ' + str(ax.index))