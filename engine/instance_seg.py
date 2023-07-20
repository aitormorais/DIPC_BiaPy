import os
import h5py
import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm

from data.post_processing.post_processing import (watershed_by_channels, calculate_optimal_mw_thresholds, voronoi_on_mask, 
                                                  remove_instance_by_circularity_central_slice, repare_large_blobs)
from data.pre_processing import create_instance_channels, create_test_instance_channels
from utils.util import save_tif
from utils.matching import matching, wrapper_matching_dataset_lazy

from engine.base_workflow import Base_Workflow

class Instance_Segmentation(Base_Workflow):
    def __init__(self, cfg, model, post_processing={}, original_test_gt_path=None):
        super().__init__(cfg, model, post_processing)

        self.original_test_gt_path = original_test_gt_path 
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            self.original_test_mask_ids = sorted(next(os.walk(self.original_test_gt_path))[2])
        self.all_matching_stats = []
        self.post_processing = post_processing

        if self.post_processing['instance_post']:
            self.all_matching_stats_post_processing = []            

        self.instance_ths = {}
        self.instance_ths['TH_BINARY_MASK'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_BINARY_MASK
        self.instance_ths['TH_CONTOUR'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_CONTOUR
        self.instance_ths['TH_FOREGROUND'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_FOREGROUND
        self.instance_ths['TH_DISTANCE'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_DISTANCE
        self.instance_ths['TH_DIST_FOREGROUND'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_DIST_FOREGROUND
        self.instance_ths['TH_POINTS'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_POINTS

        if self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_OPTIMIZE_THS and self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != "BCDv2":
            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK and os.path.isdir(self.cfg.DATA.VAL.BINARY_MASKS):
                bin_mask = self.cfg.DATA.VAL.BINARY_MASKS
            else:
                bin_mask = None
            obj = calculate_optimal_mw_thresholds(self.model, self.cfg.DATA.VAL.PATH,
                self.orig_val_gt_path, self.cfg.DATA.PATCH_SIZE, self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                self.cfg.DATA.VAL.GT_PATH, self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ, bin_mask,
                chart_dir=self.cfg.PATHS.CHARTS, verbose=self.cfg.TEST.VERBOSE)
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCD":
                self.instance_ths['TH_BINARY_MASK'], self.instance_ths['TH_CONTOUR'], self.instance_ths['TH_FOREGROUND'],\
                    self.instance_ths['TH_DISTANCE'], self.instance_ths['TH_DIST_FOREGROUND'] = obj
            else:
                self.instance_ths['TH_BINARY_MASK'], self.instance_ths['TH_CONTOUR'], self.instance_ths['TH_FOREGROUND'] = obj
            
    def instance_seg_process(self, pred, Y, filenames, f_numbers):
        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        print("Creating instances with watershed . . .")
        w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
        check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW else None
        
        w_pred = watershed_by_channels(pred, self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, ths=self.instance_ths, 
            remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW, thres_small_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ_BEFORE,
            remove_after=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_AFTER_MW, thres_small_after=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ_AFTER, 
            seed_morph_sequence=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE, seed_morph_radius=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS, 
            erode_and_dilate_foreground=self.cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND, fore_erosion_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS, 
            fore_dilation_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS, rmv_close_points=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS, 
            remove_close_points_radius=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0], resolution=self.cfg.DATA.TEST.RESOLUTION, save_dir=check_wa)

        save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
            filenames, verbose=self.cfg.TEST.VERBOSE)

        # Add extra dimension if working in 2D
        if w_pred.ndim == 2:
            w_pred = np.expand_dims(w_pred,0)

        if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
            print("Calculating matching stats . . .")

            # Need to load instance labels, as Y are binary channels used for IoU calculation
            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                del Y
                _Y = np.zeros(w_pred.shape, dtype=w_pred.dtype)
                for i in range(len(self.original_test_mask_ids)):
                    test_file = os.path.join(self.original_test_gt_path, self.original_test_mask_ids[i])
                    _Y[i] = imread(test_file).squeeze()
            else:
                test_file = os.path.join(self.original_test_gt_path, self.original_test_mask_ids[f_numbers[0]])
                _Y = imread(test_file).squeeze()

            if _Y.ndim == 2: _Y = np.expand_dims(_Y,0)

            # Convert instances to integer
            if _Y.dtype == np.float32: _Y = _Y.astype(np.uint32)
            if _Y.dtype == np.float64: _Y = _Y.astype(np.uint64)

            diff_ths_colored_img = abs(len(self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG) - len(self.cfg.TEST.MATCHING_STATS_THS))
            colored_img_ths = self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG+[-1]*diff_ths_colored_img

            results = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
            for i in range(len(results)):
                # Extract TPs, FPs and FNs from the resulting matching data structure 
                r_stats = results[i] 
                thr = r_stats['thresh']

                # TP and FN
                gt_ids = r_stats['gt_ids'][1:]
                matched_pairs = r_stats['matched_pairs']
                gt_match = [x[0] for x in matched_pairs]
                gt_unmatch = [x for x in gt_ids if x not in gt_match]
                matched_scores = list(r_stats['matched_scores'])+[0 for _ in gt_unmatch]
                pred_match = [x[1] for x in matched_pairs]+[-1 for _ in gt_unmatch]
                tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                # FPs
                pred_ids = r_stats['pred_ids'][1:]
                fp_instances = [x for x in pred_ids if x not in pred_match]
                fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                # Save csv files
                df = pd.DataFrame(zip(gt_match+gt_unmatch, pred_match, matched_scores, tag), columns =['gt_id', 'pred_id', 'iou', 'tag'])
                df = df.sort_values(by=['gt_id'])  
                df_fp = pd.DataFrame(zip(fp_instances), columns =['pred_id'])

                os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_th_{}_gt_assoc.csv'.format(thr)), index=False)
                df_fp.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_th_{}_fp.csv'.format(thr)), index=False)
                del r_stats['matched_scores']; del r_stats['matched_tps']; del r_stats['matched_pairs']; del r_stats['pred_ids']; del r_stats['gt_ids']
                print("DatasetMatching: {}".format(r_stats))

                if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                    print("Creating the image with a summary of detected points and false positives with colors . . .")
                    colored_result = np.zeros(w_pred.shape+(3,), dtype=np.uint8)

                    print("Painting TPs and FNs . . .")
                    for j in tqdm(range(len(gt_match))):
                        color = (0,255,0) if tag[j] == "TP" else (255,0,0) # Green or red
                        colored_result[np.where(_Y == gt_match[j])] = color
                    for j in tqdm(range(len(gt_unmatch))):
                        colored_result[np.where(_Y == gt_unmatch[j])] = (255,0,0) # Red

                    print("Painting FPs . . .")
                    for j in tqdm(range(len(fp_instances))):
                        colored_result[np.where(w_pred == fp_instances[j])] = (0,0,255) # Blue

                    save_tif(np.expand_dims(colored_result,0), self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                            [os.path.splitext(filenames[0])[0]+'_th_{}.tif'.format(thr)], verbose=self.cfg.TEST.VERBOSE)          
                    del colored_result
            self.all_matching_stats.append(results)


        ###################
        # Post-processing #
        ###################
        if self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
            w_pred = repare_large_blobs(w_pred, self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE)

        if self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY != -1:
            w_pred, labels, npixels, areas, circularities, diameters, comment = remove_instance_by_circularity_central_slice(w_pred, self.cfg.DATA.TEST.RESOLUTION, 
                circularity_th=self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY)

            # Save all instance stats
            size_measure = 'area' if w_pred.ndim == 2 else 'volume'
            df = pd.DataFrame(zip(np.array(labels, dtype=np.uint64), npixels, areas, circularities, diameters, comment),  
                columns=['label','npixels', size_measure, 'circularity', 'diameter', 'comment'])
            df = df.sort_values(by=['label'])   
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES, os.path.splitext(filenames[0])[0]+'_full_stats.csv'), index=False)
            # Save only remain instances stats
            df = df[df["comment"].str.contains("Strange")==False] 
            os.makedirs(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, exist_ok=True)
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, os.path.splitext(filenames[0])[0]+'_filtered_stats.csv'), index=False)
            del df

        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            w_pred = voronoi_on_mask(w_pred, pred, th=self.cfg.TEST.POST_PROCESSING.VORONOI_TH, verbose=self.cfg.TEST.VERBOSE)

        if self.post_processing['instance_post']:
            save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                filenames, verbose=self.cfg.TEST.VERBOSE)

            if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                print("Calculating matching stats after post-processing . . .")
                results = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
                
                for i in range(len(results)):
                    # Extract TPs, FPs and FNs from the resulting matching data structure 
                    r_stats = results[i] 
                    thr = r_stats['thresh']

                    # TP and FN
                    gt_ids = r_stats['gt_ids'][1:]
                    matched_pairs = r_stats['matched_pairs']
                    gt_match = [x[0] for x in matched_pairs]
                    gt_unmatch = [x for x in gt_ids if x not in gt_match]
                    matched_scores = list(r_stats['matched_scores'])+[0 for _ in gt_unmatch]
                    pred_match = [x[1] for x in matched_pairs]+[-1 for _ in gt_unmatch]
                    tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                    # FPs
                    pred_ids = r_stats['pred_ids'][1:]
                    fp_instances = [x for x in pred_ids if x not in pred_match]
                    fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                    # Save csv files
                    df = pd.DataFrame(zip(gt_match+gt_unmatch, pred_match, matched_scores, tag), columns =['gt_id', 'pred_id', 'iou', 'tag'])
                    df = df.sort_values(by=['gt_id'])  
                    df_fp = pd.DataFrame(zip(fp_instances), columns =['pred_id'])

                    os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                    df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_post-proc_th_{}_gt_assoc.csv'.format(thr)), index=False)
                    df_fp.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_post-proc_th_{}_fp.csv'.format(thr)), index=False)
                    del r_stats['matched_scores']; del r_stats['matched_tps']; del r_stats['matched_pairs']; del r_stats['pred_ids']; del r_stats['gt_ids']
                    print("DatasetMatching: {}".format(r_stats))

                    if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                        print("Creating the image with a summary of detected points and false positives with colors . . .")
                        colored_result = np.zeros(w_pred.shape+(3,), dtype=np.uint8)

                        print("Painting TPs and FNs . . .")
                        for j in tqdm(range(len(gt_match))):
                            color = (0,255,0) if tag[j] == "TP" else (255,0,0) # Green or red
                            colored_result[np.where(_Y == gt_match[j])] = color
                        for j in tqdm(range(len(gt_unmatch))):
                            colored_result[np.where(_Y == gt_unmatch[j])] = (255,0,0) # Red
                            
                        print("Painting FPs . . .")
                        for j in tqdm(range(len(fp_instances))):
                            colored_result[np.where(w_pred == fp_instances[j])] = (0,0,255) # Blue

                        save_tif(np.expand_dims(colored_result,0), self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                                [os.path.splitext(filenames[0])[0]+'_post-proc_th_{}.tif'.format(thr)], verbose=self.cfg.TEST.VERBOSE)          
                        del colored_result
                self.all_matching_stats_post_processing.append(results)

    def after_merge_patches(self, pred, Y, filenames, f_numbers):
        if not self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            self.instance_seg_process(pred, Y, filenames, f_numbers)        

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self):
        super().after_all_images()
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            print("Analysing all images as a 3D stack . . .")    
            if type(self.all_pred) is list:
                self.all_pred = np.concatenate(self.all_pred)
                self.all_gt = np.concatenate(self.all_gt) if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST) else None
            self.instance_seg_process(self.all_pred,  self.all_gt, ["3D_stack.tif"], [])

    def normalize_stats(self, image_counter): 
        super().normalize_stats(image_counter) 

        if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
            if self.cfg.TEST.MATCHING_STATS:
                self.stats['inst_stats'] = wrapper_matching_dataset_lazy(self.all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS)
                if self.post_processing['instance_post']:
                    self.stats['inst_stats_vor'] = wrapper_matching_dataset_lazy(self.all_matching_stats_post_processing, self.cfg.TEST.MATCHING_STATS_THS)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)
        super().print_post_processing_stats()

        print("Instance segmentation specific metrics:")
        if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST) :
            for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                print(self.stats['inst_stats'][i])
                if self.post_processing['instance_post']:
                    print("IoU (post-processing) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    print(self.stats['inst_stats_vor'][i])

def prepare_instance_data(cfg):
    print("###########################\n"
           "#  PREPARE INSTANCE DATA  #\n"
           "###########################\n")
    original_test_path, original_test_gt_path = None, None

    # Create selected channels for train data
    if (cfg.TRAIN.ENABLE or cfg.DATA.TEST.USE_VAL_AS_TEST) and (not os.path.isdir(cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR) or \
        not os.path.isdir(cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR)):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
        create_instance_channels(cfg)

    # Create selected channels for val data
    if cfg.TRAIN.ENABLE and not cfg.DATA.VAL.FROM_TRAIN and (not os.path.isdir(cfg.DATA.VAL.INSTANCE_CHANNELS_DIR) or \
        not os.path.isdir(cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR)):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
        create_instance_channels(cfg, data_type='val')

    # Create selected channels for test data once
    if cfg.TEST.ENABLE and not cfg.DATA.TEST.USE_VAL_AS_TEST and (not os.path.isdir(cfg.DATA.TEST.INSTANCE_CHANNELS_DIR) or \
        (not os.path.isdir(cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR) and cfg.DATA.TEST.LOAD_GT)):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
        create_test_instance_channels(cfg)

    opts = []
    if cfg.TRAIN.ENABLE:
        print("DATA.TRAIN.PATH changed from {} to {}".format(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
        print("DATA.TRAIN.GT_PATH changed from {} to {}".format(cfg.DATA.TRAIN.GT_PATH, cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR))
        opts.extend(['DATA.TRAIN.PATH', cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR,
                    'DATA.TRAIN.GT_PATH', cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR])
    if not cfg.DATA.VAL.FROM_TRAIN:
        print("DATA.VAL.PATH changed from {} to {}".format(cfg.DATA.VAL.PATH, cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
        print("DATA.VAL.GT_PATH changed from {} to {}".format(cfg.DATA.VAL.GT_PATH, cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR))
        opts.extend(['DATA.VAL.PATH', cfg.DATA.VAL.INSTANCE_CHANNELS_DIR,
                     'DATA.VAL.GT_PATH', cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR])
    if cfg.TEST.ENABLE:
        print("DATA.TEST.PATH changed from {} to {}".format(cfg.DATA.TEST.PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
        opts.extend(['DATA.TEST.PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_DIR])
        original_test_path = cfg.DATA.TEST.PATH
        if cfg.DATA.TEST.LOAD_GT:
            print("DATA.TEST.GT_PATH changed from {} to {}".format(cfg.DATA.TEST.GT_PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
            opts.extend(['DATA.TEST.GT_PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR])
        original_test_gt_path = cfg.DATA.TEST.GT_PATH
    cfg.merge_from_list(opts)

    return original_test_path, original_test_gt_path

