import random
import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
from PIL.TiffTags import TAGS

from data.pre_processing import normalize, norm_range01
from data.generators.augmentors import center_crop_single, resize_img


class test_single_data_generator(tf.keras.utils.Sequence):
    """Image data generator without data augmentation. Used only for test data.

       Parameters
       ----------
       X : Numpy 5D/4D array, optional
           Data. E.g. ``(num_of_images, z, y, x, channels)``  for ``3D`` or ``(num_of_images, y, x, channels)`` for ``2D``.

       d_path : Str, optional
           Path to load the data from.

       provide_Y: bool, optional
           Whether the ground truth has been provided or not.

       Y : Numpy 2D array, optional
           Image classes. E.g. ``(num_of_images, class)``.

       dm_path : Str, optional
           Not used here.

       dims: str, optional
           Dimension of the data. Possible options: ``2D`` or ``3D``.

       seed : int, optional
           Seed for random functions.

       instance_problem : bool, optional
           Not used here.
   
       norm_custom_mean : float, optional
           Mean of the data used to normalize.

       norm_custom_std : float, optional
           Std of the data used to normalize.

       crop_center : bool, optional
           Whether to extract a

       sample_ids :  List of ints, optional
           When cross validation is used specific training samples are passed to the generator. 
           Not used in this generator. 
    """
    def __init__(self, ndim, X=None, d_path=None, provide_Y=False, Y=None, dm_path=None, seed=42,
                 instance_problem=False, norm_custom_mean=None, norm_custom_std=None, crop_center=False, 
                 resize_shape=None, sample_ids=None):
        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")
        if provide_Y and Y is None:
            raise ValueError("'Y' must be provided")
        if crop_center and resize_shape is None:
            raise ValueError("'resize_shape' need to be provided if 'crop_center' is enabled")
        
        self.X = X
        self.Y = Y
        self.d_path = d_path
        self.provide_Y = provide_Y

        self.crop_center = crop_center
        self.resize_shape = resize_shape

        self.class_names = sorted(next(os.walk(d_path))[1])
        self.class_numbers = {}
        for i, c_name in enumerate(self.class_names):
            self.class_numbers[c_name] = i
        self.classes = {}
        if self.X is None:
            self.data_path = []
            print("Collecting data ids . . .")
            for folder in self.class_names:
                print("Analizing folder {}".format(os.path.join(d_path,folder)))
                ids = sorted(next(os.walk(os.path.join(d_path,folder)))[2])
                print("Found {} samples".format(len(ids)))
                for i in range(len(ids)):
                    self.classes[ids[i]] = folder
                    self.data_path.append(ids[i])
            self.len = len(self.data_path)
        else:
            self.len = len(X)

        self.seed = seed
        self.ndim = ndim
        self.o_indexes = np.arange(self.len)
        
        # Check if a division is required
        self.X_norm = {}
        self.X_norm['type'] = 'div'
        _, _, xnorm = self.load_sample(0)

        if norm_custom_mean is not None and norm_custom_std is not None:
            self.X_norm['type'] = 'custom'
            self.X_norm['mean'] = norm_custom_mean
            self.X_norm['std'] = norm_custom_std
        else:
            self.X_norm.update(xnorm)

        self.on_epoch_end()

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        img_class = None

        # Choose the data source
        if self.X is not None:
            img = self.X[idx]
            img = np.squeeze(img)

            if self.provide_Y:
                img_class = self.Y[idx]
        else:
            sample_id = self.data_path[idx]
            sample_class_dir = self.classes[sample_id]
            if sample_id.endswith('.npy'):
                img = np.load(os.path.join(self.data_path, sample_class_dir, sample_id))
            else:
                img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
            img_class = self.class_numbers[sample_class_dir]
            
        # Correct dimensions 
        if self.ndim == 3:
            if img.ndim == 3: 
                img = np.expand_dims(img, -1)
            else:
                min_val = min(img.shape)
                channel_pos = img.shape.index(min_val)
                if channel_pos != 3 and img.shape[channel_pos] <= 4:
                    new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                    img = img.transpose(new_pos)
        else:
            if img.ndim == 2: 
                img = np.expand_dims(img, -1) 
            else:
                if img.shape[0] <= 3: img = img.transpose((1,2,0))

        # Normalization
        xnorm = None
        if self.X_norm['type'] == 'div':
            img, xnorm = norm_range01(img)
        elif self.X_norm['type'] == 'custom':
            img = normalize(img, self.X_norm['mean'], self.X_norm['std'])
           
        img = np.expand_dims(img, 0).astype(np.float32)
 
        return img, img_class, xnorm


    def __len__(self):
        """Defines the length of the generator"""
        return self.len


    def __getitem__(self, index):
        """Generation of one sample of data.

           Parameters
           ----------
           index : int
               Sample index counter.

           Returns
           -------
           img : 3D Numpy array
               X element, for instance, an image. E.g. ``(y, x, channels)``.

           img_class : ints
               Y element, for instance, a class number.
        """

        img, img_class, norm = self.load_sample(index)
        
        if self.crop_center and img.shape[:-1] != self.resize_shape[:-1]:
            img = center_crop_single(img[0], self.resize_shape)
            img = resize_img(img, self.resize_shape[:-1])
            img = np.expand_dims(img,0)

        if norm is not None:
            self.X_norm.update(norm)           

        if self.provide_Y:
            return img, self.X_norm, img_class, None
        else:
            return img, normx

