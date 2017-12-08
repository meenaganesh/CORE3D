import numpy as np
from math import ceil
import pdb

def predict_x(x, model):
    batch_x = np.expand_dims(x, axis=0)
    batch_y = model.predict(batch_x)
    y = np.squeeze(batch_y, axis=0)
    return y

def make_prediction_img(x, target_size, predict):    
    sample_prediction = predict(x[0:target_size, 0:target_size, :])
    nb_channels = sample_prediction.shape[2]
    dtype = sample_prediction.dtype

    y = np.zeros((x.shape[0],x.shape[1],nb_channels),dtype=dtype) 

    total_window = int(ceil(x.shape[0]/target_size)*ceil(x.shape[1]/target_size))
    index = 1

    for row_begin in range(0, x.shape[0], target_size):
      for col_begin in range(0, x.shape[1], target_size):
      
        print('window: %d/%d' % (index, total_window))
        index += 1
      
        row_end = row_begin + target_size
        col_end = col_begin + target_size

        if row_end > x.shape[0]:
          row_begin = x.shape[0] - target_size 
          row_end = x.shape[0]

        if col_end > x.shape[1]:
          col_begin = x.shape[1] - target_size 
          col_end = x.shape[1]
        
        x_window = x[row_begin:row_end, col_begin:col_end, :]
        y_window = predict(x_window)
        y[row_begin:row_end, col_begin:col_end] = y_window
             
    return y


def make_legend(label_keys, label_names):
    patches = []
    for label_key, label_name in zip(label_keys, label_names):
        color = tuple(np.array(label_key) / 255.)
        patch = mpatches.Patch(
            facecolor=color, edgecolor='black', linewidth=0.5,
            label=label_name)
        patches.append(patch)
    plt.legend(handles=patches, loc='upper left',
               bbox_to_anchor=(1, 1), fontsize=4)


def plot_prediction(generator, display_all_x, display_y, display_pred,
                    file_path, is_debug=False):
    dataset = generator.dataset
    fig = plt.figure()

    nb_subplot_cols = 3
    if is_debug:
        nb_subplot_cols += len(generator.active_input_inds)

    gs = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    def plot_img(subplot_index, im, title, is_rgb=False):
        a = fig.add_subplot(gs[subplot_index])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        if is_rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)
        if subplot_index < nb_subplot_cols:
            a.set_title(title, fontsize=6)

    subplot_index = 0
    rgb_input_im = display_all_x[:, :, dataset.rgb_inds]
    plot_img(subplot_index, rgb_input_im, 'RGB', is_rgb=True)

    if is_debug:
        subplot_index += 1
        ir_im = display_all_x[:, :, dataset.ir_ind]
        plot_img(subplot_index, ir_im, 'IR')

        subplot_index += 1
        depth_im = display_all_x[:, :, dataset.depth_ind]
        plot_img(subplot_index, depth_im, 'Depth')

        subplot_index += 1
        ndvi_im = display_all_x[:, :, dataset.ndvi_ind]
        ndvi_im = (np.clip(ndvi_im, -1, 1) + 1) * 100
        plot_img(subplot_index, ndvi_im, 'NDVI')

    subplot_index += 1
    plot_img(subplot_index, display_y, 'Ground Truth',
             is_rgb=True)
             
    subplot_index += 1
    plot_img(subplot_index, display_pred, 'Prediction',
             is_rgb=True)

    make_legend(dataset.label_keys, dataset.label_names)
    plt.savefig(file_path, bbox_inches='tight', format='png', dpi=300)

    plt.close(fig)

