# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import sys
# sys.path.insert(0, './yolov5')

import numpy as np
import time
from progressbar import ProgressBar
import argparse
import os
# import platform
# import shutil
from pathlib import Path
# import cv2
import torch
# import torch.backends.cudnn as cudnn
# from yolov5.utils.datasets import LoadImages #, LoadStreams
from loadImages import LoadImages
# from yolov5.utils.general import (LOGGER, check_imshow, xyxy2xywh, increment_path)
# from my_utils.utils import annotate_boxes,  match_boxes, Detection_, xyxy2xywh_1, update_missed_box, save_results
from my_utils.utils import  match_boxes, update_missed_box, save_results
# from yolov5.utils.torch_utils import select_device  # , time_sync
# from yolov5.utils.plots import Annotator, colors
# from deep_sort.utils.parser import get_config
# from yolov5.deep_sort.configs.deepSort_configs import DS_configs
from deep_sort.deep_sort import DeepSort
# import gdown


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 deepsort root directory
# if str(ROOT) not in sys.path:
# 	sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# play = False 

# def is_docker():
#     # Is environment a Docker container?
#     return Path('/workspace').exists()  # or Path('/.dockerenv').exists()

# def is_colab():
#     # Is environment a Google Colab instance?
#     return Exception

def DS_configs():
    DEEPSORT = {'MODEL_TYPE': "resnet50",
    'MAX_DIST': 0.2, # The matching threshold. Samples with larger distance are considered an invalid match
    'MAX_IOU_DISTANCE': 0.7, # Gating threshold. Associations with cost larger than this value are disregarded.
    'MAX_AGE': 5, # Maximum number of missed misses before a track is deleted
    'N_INIT': 0, # Number of frames that a track remains in initialization phase
    'NN_BUDGET': 100} # Maximum size of the appearance descriptors gallery
    # print()
    return DEEPSORT

# def check_imshow():
#     # Check if environment supports image displays
#     try:
#         assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
#         assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
#         cv2.imshow('test', np.zeros((1, 1, 3)))
#         cv2.waitKey(1)
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#         return True
#     except Exception as e:
#         print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
#         return False

def detect(opt):
    start_time = time.time()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
		opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
		opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
		'rtsp') or source.startswith('http') or source.endswith('.txt')

    # device = select_device(opt.device)
    device = opt.device
	# initialize deepsort
    # cfg = get_config()
    # cfg.merge_from_file(opt.config_deepsort)
    cfg = DS_configs()
    # deepsort = DeepSort(deep_sort_model,
	# 					device,
	# 					max_dist=cfg.DEEPSORT.MAX_DIST,
	# 					max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
	# 					max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
	# 					)
    deepsort = DeepSort(deep_sort_model,
						device,
						max_dist=cfg['MAX_DIST'],
						max_iou_distance=cfg['MAX_IOU_DISTANCE'],
						max_age=cfg['MAX_AGE'], n_init=cfg['N_INIT'], nn_budget=cfg['NN_BUDGET'],
						)

	# Directories
    if type(yolo_model) is str:
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:
        exp_name = yolo_model[0].split(".")[0]
    else:
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
	
	# save_dir = increment_path(Path(opt.output), exist_ok=exist_ok)  # increment run if project name exists

	# Load model
	# device = select_device(device)

	# Check if environment supports image displays
    # if show_vid:
    #     show_vid = check_imshow()

	# Dataloader
	# dataset = LoadImages(source, img_size=imgsz, stride=1)
    dataset = LoadImages(source, img_size=imgsz, stride=1, reverse=opt.reverse)

	# extract what is in between the last '/' and last '.'
    # txt_file_name = source.split('/')[-1].split('.')[0]
	# txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
    out_dir = Path(opt.output)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
 
    count = 0 
    count_pred = 0
    count_hc = 0
    count_lc = 0
    count_saves = 0
    lc, hc = 0.01, opt.conf_thres
    print('\n\n')
    progress_bar = ProgressBar(len(dataset), label='Recovering labels')
    # pbar = tqdm(dataset, desc='       Recovering labels', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(pbar):
    
    for _, (path, im0s, _, s) in enumerate(dataset):
        progress_bar.update()  
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)  # im0 is bgr
        s = s.split(' ')[0]+' '+s.split(' ')[1]+' '+os.path.basename(s.split(' ')[2])
        hh,ww,_ = im0s.shape
        lc_boxes, hc_boxes = [], []
        lc_boxes_n, hc_boxes_n = [] , []
        try: 
            boxes = [x.strip('\n').split(' ') for x in list(open(path[:-4]+'.txt', 'r'))]
        except:
            if not os.path.isfile(path[:-4]+'.txt'):
                f_tmp = open(path[:-4]+'.txt','a')
                f_tmp.close()
            boxes = [x.strip('\n').split(' ') for x in list(open(path[:-4]+'.txt', 'r'))]
				
        txt_path = str(Path(opt.output)) + '/' + os.path.basename(path)[:-4]+'.txt'
			
        for line in boxes:
            cls, [bx, by, bw, bh], conf = int(line[0]), np.array(line[1:5],dtype=float), float(line[5])
            box = bx*ww, by*hh, bw*ww, bh*hh, conf, cls
            box_n = bx, by, bw, bh, conf, cls
            if lc<=conf:
                lc_boxes.append(box)
                lc_boxes_n.append(box_n)
            if conf>=hc:
                hc_boxes.append(box)
                hc_boxes_n.append(box_n)
        lc_boxes, hc_boxes = np.array(lc_boxes), np.array(hc_boxes)
  
        count_pred += len(boxes)
        count_lc += len(lc_boxes)
        count_hc += len(hc_boxes)

        if (lc_boxes is not None and len(lc_boxes)) and (hc_boxes is not None and len(hc_boxes)):
            xywhs, confs, clss = hc_boxes[:, 0:4], hc_boxes[:, 4], hc_boxes[:, 5]
            trks_confirm, trk_missed, trk_new, trks_confirm_n,trk_missed_n,trk_new_n \
       			= deepsort.update(xywhs, confs, clss, im0, hc_boxes_n) # trks_confirm, trk_missed, trk_new			

			# LOGGER.info(f'{s}Done.')
            debug = False
			
			# Match Tracks
            if len(trk_missed) > 0 and len(lc_boxes) > 0:
                unmatched_trk_missed = trk_missed
                unmatched_trk_missed_n = trk_missed_n
                if debug: print("\n\nMissed Tracks\n", trk_missed)
				
                if len(unmatched_trk_missed) > 0 and len(trks_confirm) > 0:
                    matched_trk_missed, unmatched_trk_missed, matched_trk_confirm, unmatched_trk_confirm, \
         			matched_trk_missed_n, unmatched_trk_missed_n, matched_trk_confirm_n, unmatched_trk_confirm_n, \
             		matched_idx = \
						match_boxes(unmatched_trk_missed, unmatched_trk_missed_n,\
									trks_confirm, trks_confirm_n, \
									iou_thresh = 0.70)
                    if debug: print("\nUnmatched Missed vs DeepSORT Tracks\n", unmatched_trk_missed)
						
                if len(unmatched_trk_missed) > 0 and len(trk_new) > 0:
                    matched_trk_missed, unmatched_trk_missed, matched_trk_new, unmatched_trk_new, \
         			matched_trk_missed_n, unmatched_trk_missed_n, matched_trk_new_n, unmatched_trk_new_n,\
                	matched_idx = \
                    	match_boxes(unmatched_trk_missed, unmatched_trk_missed_n,\
									trk_new, trk_new_n, \
									iou_thresh = 0.70)
                    if debug: print("\nUnmatched Missed vs DeepSORT+New Tracks\n", unmatched_trk_missed)
     
                lc_boxes[:,:4] = [deepsort._xywh_to_xyxy(box) for box in lc_boxes[:,:4]]
				# lc_boxes[:,:4] = deepsort._xywh_to_xyxy(lc_boxes[:,0:4]) 
    
                if len(unmatched_trk_missed) > 0 and len(lc_boxes) > 0:
                    matched_trk_missed, unmatched_trk_missed, matched_det_yolo_lc, unmatched_det_yolo_lc, \
         			matched_trk_missed_n, unmatched_trk_missed_n, matched_det_yolo_lc_n, unmatched_det_yolo_lc_n, \
             		matched_idx = \
						match_boxes(unmatched_trk_missed, unmatched_trk_missed_n, \
          							lc_boxes, lc_boxes_n,\
                     				iou_thresh = 0.70)
                    if debug: print("\nUnmatched Missed vs DeepSORT+New+LC Yolo Tracks\n", unmatched_trk_missed)
                    if debug: print("\nMatched Remaining Missed\n", matched_trk_missed)
                    if debug: print("\nMatched LC Yolo Tracks\n", matched_det_yolo_lc)
				
                    if not len(matched_det_yolo_lc)==0:
                        deepsort, count = update_missed_box(deepsort,matched_trk_missed, matched_trk_missed_n, \
          															 matched_det_yolo_lc, matched_det_yolo_lc_n, im0, count)
                        if save_txt:
                            save_results(matched_trk_missed,matched_trk_missed_n,txt_path,im0) # Save matched missed-lc boxes
                            count_saves = count_saves + len(matched_trk_missed)
                        play = False
						
            if save_txt:
                save_results(trks_confirm,trks_confirm_n,txt_path,im0) # Save Confirmed DeepSORT Tracks
                count_saves = count_saves + len(trks_confirm)
				
                save_results(trk_new,trk_new_n,txt_path,im0) # Save New DeepSORT Tracks
                count_saves = count_saves + len(trk_new)
				
		
        else:
            deepsort.increment_ages()
			# LOGGER.info('No detections')

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"\t\tAdded {count} additional boxes.")
    print(f"\t\tTotal predicted boxes: {count_pred}")
    print(f"\t\tTotal LC boxes: {count_lc}")
    print(f"\t\tTotal HC boxes: {count_hc}")
    print(f"\t\tTotal Boxes saved: {count_saves}")

def opts():
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-img', action='store_true', help='save image tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="/home/zafar/yolov2_demo/yolov5/deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-hc_boxes', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--reverse', action='store_true', help='augmented inference')
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    return opt

if __name__ == '__main__':
	opt = opts()
	with torch.no_grad():
		detect(opt)
