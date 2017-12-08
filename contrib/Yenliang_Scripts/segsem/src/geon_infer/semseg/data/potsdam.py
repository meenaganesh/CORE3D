from os.path import join
from geon_infer.common.utils import expand_dims

import numpy as np
from .generators import FileGenerator, TRAIN, VALIDATION, TEST
from geon_infer.common.utils import (
     save_img, load_img, get_img_size, _makedirs,
     save_numpy_array)

import pdb

POTSDAM = 'potsdam'
PROCESSED_POTSDAM = 'processed_potsdam'

class PotsdamDataset():
    def __init__(self):
        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.rgb_inds = [self.red_ind, self.green_ind, self.blue_ind]
        self.nb_channels = 6
      
        # Impervious surfaces (RGB: 255, 255, 255)
        # Building (RGB: 0, 0, 255)
        # Low vegetation (RGB: 0, 255, 255)
        # Tree (RGB: 0, 255, 0)
        # Car (RGB: 255, 255, 0)
        # Clutter/background (RGB: 255, 0, 0)
        self.label_keys = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ]

        self.nb_labels = len(self.label_keys)        
        self.label_names = [
            'Impervious',
            'Building',
            'Low vegetation',
            'Tree',
            'Car',
            'Clutter'
        ]

    @expand_dims
    def rgb_to_mask_batch(self, batch):
        mask = (batch[:, :, :, 0] == 0) & \
               (batch[:, :, :, 1] == 0) & \
               (batch[:, :, :, 2] == 0)
        mask = np.bitwise_not(mask)
        mask = np.expand_dims(mask, axis=3)
        return mask

    @expand_dims
    def rgb_to_label_batch(self, batch):
        label_batch = np.zeros(batch.shape[:-1])
        for label, key in enumerate(self.label_keys):
            mask = (batch[:, :, :, 0] == key[0]) & \
                   (batch[:, :, :, 1] == key[1]) & \
                   (batch[:, :, :, 2] == key[2])
            label_batch[mask] = label

        return np.expand_dims(label_batch, axis=3)

    @expand_dims
    def label_to_one_hot_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        nb_labels = len(self.label_keys)
        shape = np.concatenate([label_batch.shape, [nb_labels]])
        one_hot_batch = np.zeros(shape)

        for label in range(nb_labels):
            one_hot_batch[:, :, :, label][label_batch == label] = 1.
        return one_hot_batch

    @expand_dims
    def rgb_to_one_hot_batch(self, rgb_batch):
        label_batch = self.rgb_to_label_batch(rgb_batch)
        return self.label_to_one_hot_batch(label_batch)

    @expand_dims
    def label_to_rgb_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]),
                             dtype=np.uint8)
        for label, key in enumerate(self.label_keys):
            mask = label_batch == label
            rgb_batch[mask, :] = key

        return rgb_batch

    @expand_dims
    def one_hot_to_label_batch(self, one_hot_batch):
        one_hot_batch = np.argmax(one_hot_batch, axis=3)
        return np.expand_dims(one_hot_batch, axis=3)

    @expand_dims
    def one_hot_to_rgb_batch(self, one_hot_batch):
        label_batch = self.one_hot_to_label_batch(one_hot_batch)
        return self.label_to_rgb_batch(label_batch)

    def augment_channels(self, batch_x):
        red = batch_x[:, :, :, [self.red_ind]]
        ir = batch_x[:, :, :, [self.ir_ind]]
        ndvi = compute_ndvi(red, ir)
        return np.concatenate([batch_x, ndvi], axis=3)

    def get_output_file_name(self, file_ind):
        return 'top_potsdam_{}_{}_label.tif'.format(file_ind[0], file_ind[1])


class PotsdamFileGenerator(FileGenerator):
    def __init__(self, active_input_inds, train_ratio):
        self.dataset = PotsdamDataset()
        
        # training and validation
        self.file_inds = [
            (2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9), (2, 11), (2, 12), (4, 10), (5, 11),
            (6, 7), (7, 10), (7, 8)
        ]    
        # testing
        self.test_file_inds = [
            (2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15),
            (5, 13), (5, 14), (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)
        ]
        FileGenerator.__init__(self,active_input_inds, train_ratio)


class PotsdamImageFileGenerator(PotsdamFileGenerator):
    def __init__(self, datasets_path, active_input_inds,train_ratio=0.8):
        self.dataset_path = join(datasets_path, POTSDAM)
        PotsdamFileGenerator.__init__(self,active_input_inds, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        data_path = join(datasets_path, POTSDAM)
        file_path = join(
            data_path,
            '1_DSM_normalisation/dsm_potsdam_03_13_normalized_lastools.jpg')
        
        im = load_img(file_path)
        if im.shape[1] == 5999:
            im_fix = np.zeros((6000, 6000), dtype=np.uint8)
            im_fix[:, 0:-1] = im[:, :, 0]
            save_img(im_fix, file_path)

    def get_file_size(self, file_ind):
        ind0, ind1 = file_ind

        rgbir_file_path = join(
            self.dataset_path,
            '2_Ortho_RGB/top_potsdam_{}_{}_RGB.tif'.format(ind0, ind1))
        nb_rows, nb_cols = get_img_size(rgbir_file_path)
        return nb_rows, nb_cols

    def get_img(self, file_ind, window, has_y=True):
        ind0, ind1 = file_ind
        
        rgb_file_path = join(
            self.dataset_path,
            '2_Ortho_RGB/top_potsdam_{}_{}_RGB.tif'.format(ind0, ind1))
        
        depth_file_path = join(
            self.dataset_path,
            '1_DSM_normalisation/dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg'.format(ind0, ind1)) # noqa
        
        batch_y_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants/top_potsdam_{}_{}_label.tif'.format(ind0, ind1)) # noqa
        
        batch_y_no_boundary_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants_no_Boundary/top_potsdam_{}_{}_label_noBoundary.tif'.format(ind0, ind1)) # noqa
       
        rgb = load_img(rgb_file_path, window)
        channels = [rgb]

        if has_y:
            # (6000, 6000, 3)
            batch_y = load_img(batch_y_file_path, window)
            
            #(6000, 6000, 3)
            batch_y_no_boundary = load_img(batch_y_no_boundary_file_path, window)
            
            #(6000, 6000, 1)
            batch_y_depth = load_img(depth_file_path, window)
            channels.extend([batch_y, batch_y_no_boundary, batch_y_depth])
        
        # channels[0]: (10,10,3)
        # channels[1]: (10,10,3)
        # channels[2]: (10,10,3)
        # cahnnels[3]: (10,10,1)
        img = np.concatenate(channels, axis=2)
        return img    
    
    def parse_batch(self, batch, has_y=True):
        
        batch_x = batch[:, :, :, 0:3]
        batch_y = None
        batch_y_mask = None
        batch_y_depth = None

        if has_y:
            batch_y = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, 3:6])
            batch_y_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, 6:9])
            batch_y_depth = batch[:, :, :, 9:10] 
        
        # batch_x: (1, 6000, 6000, 3)
        # batch_y: (1, 6000, 6000, 6)
        # batch_y_mask: (1, 6000, 6000, 1)
        # batch_y_depth: (1, 6000, 6000, 1)
        return batch_x, batch_y, batch_y_mask, batch_y_depth

class PotsdamNumpyFileGenerator(PotsdamFileGenerator):
    def __init__(self, datasets_path, active_input_inds,train_ratio=0.8):
        
        self.raw_dataset_path = join(datasets_path, POTSDAM)
        self.dataset_path = join(datasets_path, PROCESSED_POTSDAM)
        PotsdamFileGenerator.__init__(self,active_input_inds, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_POTSDAM)
        _makedirs(proc_data_path)
        
        # active bands rgb
        generator = PotsdamImageFileGenerator(
                    datasets_path, [0, 1, 2])
        dataset = generator.dataset

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, model_type=None, batch_size=1, shuffle=False, augment=False,
                normalize=False, eval_mode=True)
            
            for (batch_x, batch_y, all_batch_x, batch_y_mask, batch_y_depth,
                    batch_file_inds) in gen:

                file_ind = batch_file_inds[0]
                x = np.squeeze(batch_x, axis=0)
                channels = [x]
 
                if batch_y is not None:
                    y = np.squeeze(batch_y, axis=0)
                    y = dataset.one_hot_to_label_batch(y)
                    y_mask = np.squeeze(batch_y_mask, axis=0)
                    y_depth = np.squeeze(batch_y_depth, axis=0)
                    channels.extend([y, y_mask, y_depth])
                channels = np.concatenate(channels, axis=2)

                ind0, ind1 = file_ind
                file_name = '{}_{}'.format(ind0, ind1)
                save_numpy_array(
                join(proc_data_path, file_name), channels)

                channels = None
                batch_x = x = None
                batch_y = y = None
                batch_y_mask = y_mask = None
                batch_y_depth = y_depth = None
  
        _preprocess(TRAIN)
        _preprocess(VALIDATION)
        _preprocess(TEST)

    def get_file_path(self, file_ind):
        ind0, ind1 = file_ind
        return join(self.dataset_path, '{}_{}.npy'.format(ind0, ind1))

    def get_file_size(self, file_ind):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_img(self, file_ind, window, has_y=True):
        file_path = self.get_file_path(file_ind) 
        im = np.load(file_path, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        img = im[row_begin:row_end, col_begin:col_end, :]

        return img

    def parse_batch(self, batch, has_y=True):
        batch_x = batch[:, :, :, 0:3]
        batch_y = None
        batch_y_mask = None
        batch_y_depth = None

        if has_y:
            batch_y = self.dataset.label_to_one_hot_batch(batch[:, :, :, 3:4])
            batch_y_mask = batch[:, :, :, 4:5]
            batch_y_depth = batch[:,:, :, 5:6]

        return batch_x, batch_y, batch_y_mask, batch_y_depth
