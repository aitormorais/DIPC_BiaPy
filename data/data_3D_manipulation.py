import math
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.util import load_3d_images_from_dir


def load_and_prepare_3D_data(train_path, train_gt_path, cross_val=False, cross_val_nsplits=5, cross_val_fold=1, 
    val_split=0.1, seed=0, shuffle_val=True, crop_shape=(80, 80, 80, 1), y_upscaling=1, random_crops_in_DA=False, 
    ov=(0,0,0), padding=(0,0,0), minimum_foreground_perc=-1, reflect_to_complete_shape=False):
    """
    Load train and validation images from the given paths to create 3D data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_gt_path : str
        Path to the training data masks.

    cross_val : bool, optional
        Whether to use cross validation or not. 

    cross_val_nsplits : int, optional
        Number of folds for the cross validation. 
    
    cross_val_fold : int, optional
        Number of the fold to be used as validation. 

    val_split : float, optional
        ``%`` of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    crop_shape : 4D tuple
        Shape of the train subvolumes to create. E.g. ``(z, y, x, channels)``.

    y_upscaling : int, optional
        Upscaling to be done when loading Y data. User for super-resolution workflow.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

    ov : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
        or ``99%`` of overlap. E. g. ``(z, y, x)``.

    padding : Tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    minimum_foreground_perc : float, optional
        Minimum percetnage of foreground that a sample need to have no not be discarded. 

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    self_supervised_args : dict, optional
        Arguments to create ground truth data for self-supervised workflow. 

    Returns
    -------
    X_train : 5D Numpy array
        Train images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_train : 5D Numpy array
        Train images' mask. E.g. ``(num_of_images, z, y, x, channels)``.

    X_val : 5D Numpy array, optional
        Validation images (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    Y_val : 5D Numpy array, optional
        Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    filenames : List of str
        Loaded train filenames.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Case where we need to load the data and creating a validation split
        train_path = "data/train/x"
        train_gt_path = "data/train/y"

        # Train data is (15, 91, 1024, 1024) where (number_of_images, z, y, x), so each image shape should be this:
        img_train_shape = (91, 1024, 1024, 1)
        # 3D subvolume shape needed
        train_3d_shape = (40, 256, 256, 1)

        X_train, Y_train, X_val,
        Y_val, filenames = load_and_prepare_3D_data_v2(train_path, train_gt_path, train_3d_shape,
                                                        val_split=0.1, shuffle_val=True, ov=(0,0,0))

        # The function will print the shapes of the generated arrays. In this example:
        #     *** Loaded train data shape is: (315, 40, 256, 256, 1)
        #     *** Loaded train mask shape is: (315, 40, 256, 256, 1)
        #     *** Loaded validation data shape is: (35, 40, 256, 256, 1)
        #     *** Loaded validation mask shape is: (35, 40, 256, 256, 1)
        #
    """

    print("### LOAD ###")

    # Disable crops when random_crops_in_DA is selected
    if random_crops_in_DA:
        crop = False  
    else:
        if cross_val:
            crop = False
            # Delay the crop to be made after cross validation
            delay_crop = True  
        else:
            crop = True
            delay_crop = False  

    # Check validation
    if val_split > 0 or cross_val:
        create_val = True  
    else:
        create_val = False

    print("0) Loading train images . . .")
    X_train, _, _, t_filenames = load_3d_images_from_dir(train_path, crop=crop, crop_shape=crop_shape,
        overlap=ov, padding=padding, return_filenames=True, reflect_to_complete_shape=reflect_to_complete_shape)

    if train_gt_path is not None:
        print("1) Loading train masks . . .")
        scrop = (crop_shape[0], crop_shape[1]*y_upscaling, crop_shape[2]*y_upscaling, crop_shape[3])
        Y_train, _, _ = load_3d_images_from_dir(train_gt_path, crop=crop, crop_shape=scrop, overlap=ov,
            padding=padding, reflect_to_complete_shape=reflect_to_complete_shape, check_channel=False)
    else:
        Y_train = np.zeros(X_train.shape, dtype=np.float32) # Fake mask val

    if isinstance(X_train, list):
        raise NotImplementedError("If you arrived here means that your images are not all of the same shape, and you "
                                  "select DATA.EXTRACT_RANDOM_PATCH = True, so no crops are made to ensure all images "
                                  "have the same shape. Please, crop them into your DATA.PATCH_SIZE and run again (you "
                                  "can use one of the script from here to crop: https://github.com/danifranco/BiaPy/tree/master/utils/scripts)")

    # Discard images that do not surpass the foreground percentage threshold imposed 
    if minimum_foreground_perc != -1:
        print("Data that do not have {}% of foreground is discarded".format(minimum_foreground_perc))

        X_train_keep = []
        Y_train_keep = []
        are_lists = True if type(Y_train) is list else False

        samples_discarded = 0
        for i in tqdm(range(len(Y_train)), leave=False):
            labels, npixels = np.unique((Y_train[i]>0).astype(np.uint8), return_counts=True)

            total_pixels = 1
            for val in list(Y_train[i].shape):
                total_pixels *= val
            
            discard = False
            if len(labels) == 1:
                discard = True
            else:
                if (sum(npixels[1:]/total_pixels)) < minimum_foreground_perc:
                    discard = True

            if discard:
                samples_discarded += 1
            else:
                if are_lists:
                    X_train_keep.append(X_train[i])
                    Y_train_keep.append(Y_train[i])
                else:
                    X_train_keep.append(np.expand_dims(X_train[i],0))
                    Y_train_keep.append(np.expand_dims(Y_train[i],0))
        del X_train, Y_train
        
        if not are_lists:
            X_train_keep = np.concatenate(X_train_keep)
            Y_train_keep = np.concatenate(Y_train_keep)
        
        # Rename 
        X_train, Y_train = X_train_keep, Y_train_keep 
        del X_train_keep, Y_train_keep 

        print("{} samples discarded!".format(samples_discarded)) 
        if type(Y_train) is not list:      
            print("*** Remaining data shape is {}".format(X_train.shape))
            if X_train.shape[0] <= 1 and create_val: 
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                "Please, decrease the percentage to be more permissive")
        else:
            print("*** Remaining data shape is {}".format((len(X_train),)+X_train[0].shape[1:]))
            if len(X_train) <= 1 and create_val:
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                "Please, decrease the percentage to be more permissive")

    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        if not cross_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val,
                random_state=seed)
            fold = 1
            train_index, test_index = None, None

            for t_index, te_index in skf.split(np.zeros(len(X_train)), np.zeros(len(Y_train))):
                if cross_val_fold == fold:
                    X_train, X_val = X_train[t_index], X_train[te_index]
                    Y_train, Y_val = Y_train[t_index], Y_train[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold+= 1
            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

            # Then crop after cross validation
            if delay_crop:
                # X_train
                data = []
                for img_num in range(len(X_train)):
                    if X_train[img_num].shape != crop_shape[:3]+(X_train[img_num].shape[-1],):
                        img = X_train[img_num]
                        img = crop_3D_data_with_overlap(X_train[img_num][0] if isinstance(X_train, list) else X_train[img_num], 
                            crop_shape[:3]+(X_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_train = np.concatenate(data)
                del data

                # Y_train
                data_mask = []
                scrop = (crop_shape[0], crop_shape[1]*y_upscaling, crop_shape[2]*y_upscaling, crop_shape[3])
                for img_num in range(len(Y_train)):
                    if Y_train[img_num].shape != scrop[:3]+(Y_train[img_num].shape[-1],):
                        img = Y_train[img_num]
                        img = crop_3D_data_with_overlap(Y_train[img_num][0] if isinstance(Y_train, list) else Y_train[img_num],
                            scrop[:3]+(Y_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data_mask.append(img)
                Y_train = np.concatenate(data_mask)
                del data_mask
                
                # X_val
                data = []
                for img_num in range(len(X_val)):
                    if X_val[img_num].shape != crop_shape[:3]+(X_val[img_num].shape[-1],):
                        img = X_val[img_num]
                        img = crop_3D_data_with_overlap(X_val[img_num][0] if isinstance(X_val, list) else X_val[img_num], 
                            crop_shape[:3]+(X_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_val = np.concatenate(data)
                del data

                # Y_val
                data_mask = []
                scrop = (crop_shape[0], crop_shape[1]*y_upscaling, crop_shape[2]*y_upscaling, crop_shape[3])
                for img_num in range(len(Y_val)):
                    if Y_val[img_num].shape != scrop[:3]+(Y_val[img_num].shape[-1],):
                        img = Y_val[img_num]
                        img = crop_3D_data_with_overlap(Y_val[img_num][0] if isinstance(Y_val, list) else Y_val[img_num],
                            scrop[:3]+(Y_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data_mask.append(img)
                Y_val = np.concatenate(data_mask)
                del data_mask

    # Convert the original volumes as they were a unique subvolume
    if random_crops_in_DA and X_train.ndim == 4:
        X_train = np.expand_dims(X_train, axis=0)
        Y_train = np.expand_dims(Y_train, axis=0)
        if create_val:
            X_val = np.expand_dims(X_val, axis=0)
            Y_val = np.expand_dims(Y_val, axis=0)

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded train mask shape is: {}".format(Y_train.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        print("*** Loaded validation mask shape is: {}".format(Y_val.shape))
        if not cross_val:
            return X_train, Y_train, X_val, Y_val, t_filenames
        else:
            return X_train, Y_train, X_val, Y_val, t_filenames, test_index
    else:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded train mask shape is: {}".format(Y_train.shape))
        return X_train, Y_train, t_filenames


def crop_3D_data_with_overlap(data, vol_shape, data_mask=None, overlap=(0,0,0), padding=(0,0,0), verbose=True,
    median_padding=False):
    """Crop 3D data into smaller volumes with a defined overlap. The opposite function is :func:`~merge_3D_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, y, x, channels)``.

       vol_shape : 4D int tuple
           Shape of the volumes to create. E.g. ``(z, y, x, channels)``.

       data_mask : 4D Numpy array, optional
            Data mask to crop. E.g. ``(num_of_images, y, x, channels)``.

       overlap : Tuple of 3 floats, optional
            Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
            or ``99%`` of overlap. E.g. ``(z, y, x)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       median_padding : bool, optional
           If ``True`` the padding value is the median value. If ``False``, the added values are zeroes.

       Returns
       -------
       cropped_data : 5D Numpy array
           Cropped image data. E.g. ``(vol_number, z, y, x, channels)``.

       cropped_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(vol_number, z, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Following the example introduced in load_and_prepare_3D_data function, the cropping of a volume with shape
           # (165, 1024, 765) should be done by the following call:
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))
           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train,
                                                        overlap=(0.5,0.5,0.5))
           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (2600, 80, 80, 80, 1)

       A visual explanation of the process:

       .. image:: ../../img/crop_3D_ov.png
           :width: 80%
           :align: center

       Note: this image do not respect the proportions.
       ::

           # EXAMPLE 2
           # Same data crop but without overlap

           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0,0,0))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (390, 80, 80, 80, 1)
           #
           # Notice how differs the amount of subvolumes created compared to the first example

           #EXAMPLE 2
           #In the same way, if the addition of (64,64,64) padding is required, the call should be done as shown:
           X_train, Y_train = crop_3D_data_with_overlap(
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5), padding=(64,64,64))
    """

    if verbose:
        print("### 3D-OV-CROP ###")
        print("Cropping {} images into {} with overlapping . . .".format(data.shape, vol_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if data.ndim != 4:
        raise ValueError("data expected to be 4 dimensional, given {}".format(data.shape))
    if data_mask is not None:
        if data_mask.ndim != 4:
            raise ValueError("data_mask expected to be 4 dimensional, given {}".format(data_mask.shape))
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))
    if len(vol_shape) != 4:
        raise ValueError("vol_shape expected to be of length 4, given {}".format(vol_shape))
    if vol_shape[0] > data.shape[0]:
        raise ValueError("'vol_shape[0]' {} greater than {}".format(vol_shape[0], data.shape[0]))
    if vol_shape[1] > data.shape[1]:
        raise ValueError("'vol_shape[1]' {} greater than {}".format(vol_shape[1], data.shape[1]))
    if vol_shape[2] > data.shape[2]:
        raise ValueError("'vol_shape[2]' {} greater than {}".format(vol_shape[2], data.shape[2]))
    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0) or (overlap[2] >= 1 or overlap[2] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")
    for i,p in enumerate(padding):
        if p >= vol_shape[i]//2:
            raise ValueError("'Padding' can not be greater than the half of 'vol_shape'. Max value for this {} input shape is {}"
                             .format(data.shape, [(vol_shape[0]//2)-1,(vol_shape[1]//2)-1,(vol_shape[2]//2)-1]))

    padded_data = np.pad(data,((padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2]),(0,0)), 'reflect')
    if data_mask is not None:
        padded_data_mask = np.pad(data_mask,((padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2]),(0,0)), 'reflect')
    if median_padding:
    	padded_data[0:padding[0], :, :, :] = np.median(data[0, :, :, :])
    	padded_data[padding[0]+data.shape[0]:2*padding[0]+data.shape[0], :, :, :] = np.median(data[-1, :, :, :])
    	padded_data[:, 0:padding[1], :, :] = np.median(data[:, 0, :, :])
    	padded_data[:, padding[1]+data.shape[1]:2*padding[1]+data.shape[0], :, :] = np.median(data[:, -1, :, :])
    	padded_data[:, :, 0:padding[2], :] = np.median(data[:, :, 0, :])
    	padded_data[ :, :, padding[2]+data.shape[2]:2*padding[2]+data.shape[2], :] = np.median(data[:, :, -1, :])
    padded_vol_shape = vol_shape

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1-overlap[2]

    # Z
    step_z = int((vol_shape[0]-padding[0]*2)*overlap_z)
    vols_per_z = math.ceil(data.shape[0]/step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z-1)*step_z)+vol_shape[0])-padded_data.shape[0]
    ovz_per_block = last_z//(vols_per_z-1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block*(vols_per_z-1)

    # Y
    step_y = int((vol_shape[1]-padding[1]*2)*overlap_y)
    vols_per_y = math.ceil(data.shape[1]/step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y-1)*step_y)+vol_shape[1])-padded_data.shape[1]
    ovy_per_block = last_y//(vols_per_y-1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block*(vols_per_y-1)

    # X
    step_x = int((vol_shape[2]-padding[2]*2)*overlap_x)
    vols_per_x = math.ceil(data.shape[2]/step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x-1)*step_x)+vol_shape[2])-padded_data.shape[2]
    ovx_per_block = last_x//(vols_per_x-1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block*(vols_per_x-1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block/(vol_shape[0]-padding[0]*2)
    real_ov_y = ovy_per_block/(vol_shape[1]-padding[1]*2)
    real_ov_x = ovx_per_block/(vol_shape[2]-padding[2]*2)
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z,real_ov_y,real_ov_x)))
        print("Real overlapping (pixels): {}".format(((vol_shape[0]-padding[0]*2)*real_ov_z,
              (vol_shape[1]-padding[1]*2)*real_ov_y,(vol_shape[2]-padding[2]*2)*real_ov_x)))
        print("{} patches per (z,y,x) axis".format((vols_per_z,vols_per_x,vols_per_y)))
    
    total_vol = vols_per_z*vols_per_y*vols_per_x
    cropped_data = np.zeros((total_vol,) + padded_vol_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,) + padded_vol_shape[:3]+(data_mask.shape[-1],), dtype=data_mask.dtype)

    c = 0
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z*step_z+vol_shape[0]) < padded_data.shape[0] else last_z
                d_y = 0 if (y*step_y+vol_shape[1]) < padded_data.shape[1] else last_y
                d_x = 0 if (x*step_x+vol_shape[2]) < padded_data.shape[2] else last_x

                cropped_data[c] = padded_data[z*step_z-d_z:z*step_z+vol_shape[0]-d_z,
                                              y*step_y-d_y:y*step_y+vol_shape[1]-d_y,
                                              x*step_x-d_x:x*step_x+vol_shape[2]-d_x]
                if data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[z*step_z-d_z:(z*step_z)+vol_shape[0]-d_z,
                                                            y*step_y-d_y:y*step_y+vol_shape[1]-d_y,
                                                            x*step_x-d_x:x*step_x+vol_shape[2]-d_x]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END 3D-OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_3D_data_with_overlap(data, orig_vol_shape, data_mask=None, overlap=(0,0,0), padding=(0,0,0), verbose=True):
    """Merge 3D subvolumes in a 3D volume with a defined overlap.

       The opposite function is :func:`~crop_3D_data_with_overlap`.

       Parameters
       ----------
       data : 5D Numpy array
           Data to crop. E.g. ``(volume_number, z, y, x, channels)``.

       orig_vol_shape : 4D int tuple
           Shape of the volumes to create.

       data_mask : 4D Numpy array, optional
           Data mask to crop. E.g. ``(volume_number, z, y, x, channels)``.

       overlap : Tuple of 3 floats, optional
            Amount of minimum overlap on x, y and z dimensions. Should be the same as used in
            :func:`~crop_3D_data_with_overlap`. The values must be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of
            overlap. E.g. ``(z, y, x)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       Returns
       -------
       merged_data : 4D Numpy array
           Cropped image data. E.g. ``(num_of_images, y, x, channels)``.

       merged_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(num_of_images, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Following the example introduced in crop_3D_data_with_overlap function, the merge after the cropping
           # should be done as follows:

           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))
           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)

           # EXAMPLE 2
           # In the same way, if no overlap in cropping was selected, the merge call
           # should be as follows:

           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0,0,0))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)

           # EXAMPLE 3
           # On the contrary, if no overlap in cropping was selected but a padding of shape
           # (64,64,64) is needed, the merge call should be as follows:

           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0,0,0),
               padding=(64,64,64))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)
    """
    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))

    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0) or (overlap[2] >= 1 or overlap[2] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    if verbose:
        print("### MERGE-3D-OV-CROP ###")
        print("Merging {} images into {} with overlapping . . .".format(data.shape, orig_vol_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    # Remove the padding
    pad_input_shape = data.shape
    data = data[:, padding[0]:data.shape[1]-padding[0],
                padding[1]:data.shape[2]-padding[1],
                padding[2]:data.shape[3]-padding[2], :]

    merged_data = np.zeros((orig_vol_shape), dtype=np.float32)
    if data_mask is not None:
        data_mask = data_mask[:, padding[0]:data_mask.shape[1]-padding[0],
                              padding[1]:data_mask.shape[2]-padding[1],
                              padding[2]:data_mask.shape[3]-padding[2], :]
        merged_data_mask = np.zeros(orig_vol_shape[:3]+(data_mask.shape[-1],), dtype=np.float32)
    ov_map_counter = np.zeros((orig_vol_shape), dtype=np.uint16)

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1-overlap[2]
    
    padded_vol_shape = [orig_vol_shape[0]+2*padding[0], orig_vol_shape[1]+2*padding[1], orig_vol_shape[2]+2*padding[2]]

    # Z
    step_z = int((pad_input_shape[1]-padding[0]*2)*overlap_z)
    vols_per_z = math.ceil(orig_vol_shape[0]/step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z-1)*step_z)+pad_input_shape[1])-padded_vol_shape[0]
    ovz_per_block = last_z//(vols_per_z-1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block*(vols_per_z-1)

    # Y
    step_y = int((pad_input_shape[2]-padding[1]*2)*overlap_y)
    vols_per_y = math.ceil(orig_vol_shape[1]/step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y-1)*step_y)+pad_input_shape[2])-padded_vol_shape[1]
    ovy_per_block = last_y//(vols_per_y-1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block*(vols_per_y-1)

    # X
    step_x = int((pad_input_shape[3]-padding[2]*2)*overlap_x)
    vols_per_x = math.ceil(orig_vol_shape[2]/step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x-1)*step_x)+pad_input_shape[3])-padded_vol_shape[2]
    ovx_per_block = last_x//(vols_per_x-1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block*(vols_per_x-1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block/(pad_input_shape[1]-padding[0]*2) 
    real_ov_y = ovy_per_block/(pad_input_shape[2]-padding[1]*2) 
    real_ov_x = ovx_per_block/(pad_input_shape[3]-padding[2]*2) 

    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z,real_ov_y,real_ov_x)))
        print("Real overlapping (pixels): {}".format(((pad_input_shape[1]-padding[0]*2)*real_ov_z,
            (pad_input_shape[2]-padding[1]*2)*real_ov_y,(pad_input_shape[3]-padding[2]*2)*real_ov_x)))
        print("{} patches per (z,y,x) axis".format((vols_per_z,vols_per_x,vols_per_y)))

    c = 0
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z*step_z+data.shape[1]) < orig_vol_shape[0] else last_z
                d_y = 0 if (y*step_y+data.shape[2]) < orig_vol_shape[1] else last_y
                d_x = 0 if (x*step_x+data.shape[3]) < orig_vol_shape[2] else last_x

                merged_data[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                            y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                            x*step_x-d_x:x*step_x+data.shape[3]-d_x] += data[c]

                if data_mask is not None:
                    merged_data_mask[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                                     y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                                     x*step_x-d_x:x*step_x+data.shape[3]-d_x] += data_mask[c]

                ov_map_counter[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                               y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                               x*step_x-d_x:x*step_x+data.shape[3]-d_x] += 1
                c += 1

    merged_data = np.true_divide(merged_data, ov_map_counter).astype(data.dtype)

    if verbose:
        print("**** New data shape is: {}".format(merged_data.shape))
        print("### END MERGE-3D-OV-CROP ###")

    if data_mask is not None:
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter).astype(data_mask.dtype)
        return merged_data, merged_data_mask
    else:
        return merged_data


def load_3d_data_classification(cfg, test=False):
    """Load 3D data to train classification methods.

       Parameters
       ----------
       test : bool, optional
           To load test data instead of train/validation.

       Returns
       -------
       X_data : 5D Numpy array
           Train/test images. E.g. ``(num_of_images, z, y, x, channels)``.

       Y_data : 1D Numpy array
           Train/test images' classes. E.g. ``(num_of_images)``.

       ids : List of str
           Filenames loaded.
       
       class_names : List of str
           Class names extracted from directory names.

       X_val : 4D Numpy array, optional
           Validation images. E.g. ``(num_of_images, z, y, x, channels)``.

       Y_val : 1D Numpy array, optional
           Validation images' classes. E.g. ``(num_of_images)``.
    """

    print("### LOAD ###")
    if not test:
        path = cfg.DATA.TRAIN.PATH
    else:
        path = cfg.DATA.TEST.PATH

    all_ids = []
    if not test:
        if not cfg.DATA.VAL.CROSS_VAL:
            X_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_train.npy')
            Y_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_train.npy')
            X_val_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_val.npy')
            Y_val_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_val.npy')
        else:
            f_info = str(cfg.DATA.VAL.CROSS_VAL_FOLD)+'of'+str(cfg.DATA.VAL.CROSS_VAL_NFOLD)
            X_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_train'+f_info+'.npy')
            Y_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_train'+f_info+'.npy')
            X_val_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_val'+f_info+'.npy')
            Y_val_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_val'+f_info+'.npy')
    else:
        if not cfg.DATA.TEST.USE_VAL_AS_TEST:
            X_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_test.npy')
            Y_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_test.npy')
        else:
            f_info = str(cfg.DATA.VAL.CROSS_VAL_FOLD)+'of'+str(cfg.DATA.VAL.CROSS_VAL_NFOLD)
            X_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'X_val'+f_info+'.npy')
            Y_data_npy_file = os.path.join(path, '../npy_data_for_classification', 'Y_val'+f_info+'.npy')

    class_names = sorted(next(os.walk(path))[1])
    if not os.path.exists(X_data_npy_file):
        print("Seems to be the first run as no data is prepared. Creating .npy files: {}".format(X_data_npy_file))
        if not test:
            print("## TRAIN ##")
        else:
            print("## TEST ##")
        X_data, Y_data = [], []
        for c_num, folder in enumerate(class_names):
            print("Analizing folder {}".format(os.path.join(path,folder)))
            ids = sorted(next(os.walk(os.path.join(path,folder)))[2])
            all_ids.append(ids)
            print("Found {} samples".format(len(ids)))
            class_X_data, class_Y_data = [], []
            for i in tqdm(range(len(ids)), leave=False):
                img = imread(os.path.join(path, folder, ids[i]))
                img = np.squeeze(img)

                if img.ndim < 3:
                    raise ValueError("Read image seems to be 2D: {}. Path: {}".format(img.shape, os.path.join(data_dir, id_)))
                        
                if img.ndim == 3: 
                    img = np.expand_dims(img, -1)
                else:
                    min_val = min(img.shape)
                    channel_pos = img.shape.index(min_val)
                    if channel_pos != 3 and img.shape[channel_pos] <= 4:
                        new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                        img = img.transpose(new_pos)
                img = np.expand_dims(img, 0)

                if cfg.DATA.PATCH_SIZE[-1] != img.shape[-1]:
                    raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                        "Please, check the channels of the images!".format(cfg.DATA.PATCH_SIZE[-1], img.shape[-1]))

                class_X_data.append(img)
                class_Y_data.append(np.expand_dims(np.array(c_num),0).astype(np.uint8))

            class_X_data = np.concatenate(class_X_data, 0)
            class_Y_data = np.concatenate(class_Y_data, 0)
            X_data.append(class_X_data)
            Y_data.append(class_Y_data)

        # Fuse all data
        X_data = np.concatenate(X_data, 0)
        Y_data = np.concatenate(Y_data, 0)
        Y_data = np.squeeze(Y_data)

        os.makedirs(os.path.join(path, '../npy_data_for_classification'), exist_ok=True)
        if not test:
            print("## VAL ##")
            X_val, Y_val = [], []
            if cfg.DATA.VAL.FROM_TRAIN:
                if cfg.DATA.VAL.CROSS_VAL: 
                    skf = StratifiedKFold(n_splits=cfg.DATA.VAL.CROSS_VAL_NFOLD, shuffle=cfg.DATA.VAL.RANDOM,
                        random_state=cfg.SYSTEM.SEED)
                    f_num = 1
                    for train_index, test_index in skf.split(X_data, Y_data):
                        if cfg.DATA.VAL.CROSS_VAL_FOLD == f_num:
                            X_data, X_val = X_data[train_index], X_data[test_index]
                            Y_data, Y_val = Y_data[train_index], Y_data[test_index]
                            break
                        f_num+= 1
                else:
                    X_data, X_val, Y_data, Y_val = train_test_split(X_data, Y_data, test_size=cfg.DATA.VAL.SPLIT_TRAIN,
                        shuffle=cfg.DATA.VAL.RANDOM, random_state=cfg.SYSTEM.SEED)
            else:
                path_val = cfg.DATA.VAL.PATH
                class_names = sorted(next(os.walk(path_val))[1])
                for c_num, folder in enumerate(class_names):
                    print("Analizing folder {}".format(os.path.join(path_val, folder)))
                    ids = sorted(next(os.walk(os.path.join(path_val,folder)))[2])
                    print("Found {} samples".format(len(ids)))
                    class_X_data, class_Y_data = [], []
                    for i in tqdm(range(len(ids)), leave=False):
                        img = imread(os.path.join(path_val, folder, ids[i]))
                        img = np.squeeze(img)

                        if img.ndim < 3:
                            raise ValueError("Read image seems to be 2D: {}. Path: {}".format(img.shape, os.path.join(data_dir, id_)))
                            
                        if img.ndim == 3: 
                            img = np.expand_dims(img, -1)
                        else:
                            min_val = min(img.shape)
                            channel_pos = img.shape.index(min_val)
                            if channel_pos != 3 and img.shape[channel_pos] <= 4:
                                new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                                img = img.transpose(new_pos)
                        img = np.expand_dims(img, 0)

                        if cfg.DATA.PATCH_SIZE[-1] != img.shape[-1]:
                            raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                                "Please, check the channels of the images!".format(cfg.DATA.PATCH_SIZE[-1], img.shape[-1]))
                        
                        class_X_data.append(img)
                        class_Y_data.append(np.expand_dims(np.array(c_num),0).astype(np.uint8))

                    class_X_data = np.concatenate(class_X_data, 0)
                    class_Y_data = np.concatenate(class_Y_data, 0)
                    X_val.append(class_X_data)
                    Y_val.append(class_Y_data)

                # Fuse all data
                X_val = np.concatenate(X_val, 0)
                Y_val = np.concatenate(Y_val, 0)
                Y_val = np.squeeze(Y_val)

            np.save(X_val_npy_file, X_val)
            np.save(Y_val_npy_file, Y_val)

        np.save(X_data_npy_file, X_data)
        np.save(Y_data_npy_file, Y_data)
    else:
        X_data = np.load(X_data_npy_file)
        Y_data = np.load(Y_data_npy_file)
        if not test:
            X_val = np.load(X_val_npy_file)
            Y_val = np.load(Y_val_npy_file)

        for c_num, folder in enumerate(class_names):
            ids = sorted(next(os.walk(os.path.join(path, folder)))[2])
            all_ids.append(ids)
    
    all_ids = np.concatenate(all_ids)
    if not test:
        print("*** Loaded train data shape is: {}".format(X_data.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        print("### END LOAD ###")
        return X_data, Y_data, X_val, Y_val
    else:
        print("*** Loaded test data shape is: {}".format(X_data.shape))
        return X_data, Y_data, all_ids, class_names
