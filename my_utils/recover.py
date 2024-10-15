import numpy as np
import os
from pathlib import Path
import sys

from tqdm import tqdm
from copy import deepcopy
import cv2
import torch

sys.path.append("..")
sys.path.append("../yolov5")

import bbox as cal_bboxiou
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from my_utils.utils import annotate_boxes,  match_boxes, Detection_, xyxy2xywh_1, update_missed_box, save_results
from my_utils.recover import annotate_boxes,  match_boxes, Detection_, xyxy2xywh_1, update_missed_box, save_results
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.general import xyxy2xywh, xywh2xyxy

class Recover:
	def __init__(self, reverse, deep_sort_model, config_deepsort, device):
		
		# initialize deepsort
		deep_sort_model = deep_sort_model
		cfg = get_config()
		cfg.merge_from_file(config_deepsort)
		self.deepsort = DeepSort(deep_sort_model,
							device,
							max_dist=cfg.DEEPSORT.MAX_DIST,
							max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
							max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
							)
	
	def update(self, pred, pred_lc, pred_hc, im_path,names):
		im0 = cv2.imread(im_path)
		annotator = Annotator(im0, line_width=2, pil=not ascii)
  
		xywhs, confs, clss = xyxy2xywh(pred_hc[:, 0:4]), pred_hc[:, 4], pred_hc[:, 5]
		trks_confirm, trk_missed, trk_new = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0, [])
		ntc,ntm,ntn = len(trks_confirm), len(trk_missed), len(trk_new)		
		print(f'{os.path.basename(im_path)}: Tracks - C:{ntc}, M:{ntm}, N:{ntn}')
		data = [ntc, ntm, ntn, trks_confirm, trk_missed, trk_new]
		trks_c, trks_m, trks_n = trks_confirm, trk_missed, trk_new
		if len(trks_c): trks_c[:,0:4] = xywh2xyxy(trks_c[:,0:4])
		if len(trks_c): annotate_boxes(annotator, trks_c, names, color=(0,255,0))
		if len(trks_m): trks_m[:,0:4] = xywh2xyxy(trks_m[:,0:4])
		if len(trks_m): annotate_boxes(annotator, trks_m, names, color=(0,0,255))
		if len(trks_n): trks_n[:,0:4] = xywh2xyxy(trks_n[:,0:4])
		if len(trks_n): annotate_boxes(annotator, trks_n, names, color=(255,0,0))
		im0 = annotator.result()
		new_width = 1000
		dim = (int(new_width * (im0.shape[1]/im0.shape[0])), int(new_width))
		resized_same_img = cv2.resize(im0, dim, interpolation = cv2.INTER_AREA)
		
		return pred, data, resized_same_img

		# return pred, len(trks_confirm), len(trk_missed), len(trk_new)
  
  	
	def read_generated_predictions(self, pred, im_path, shape, saved_path, device):
		gn = torch.tensor(shape)[[1, 0, 1, 0]]
		ww,hh = shape
		# pred_np = pred.cpu().detach().numpy()
		label_path = os.path.join( saved_path, os.path.basename(im_path).replace('jpg','txt') )
		labels = np.array([x.strip(' \n').split(' ') for x in open(label_path,'r').readlines()  if not x=='' ],dtype=np.float32)
		if len(labels)		:
			labels_scaled = deepcopy(labels)
			labels_scaled[:,5] = labels[:,0]
			labels_scaled[:,4] = [max(x,0.51) for x in labels[:,5]]
			labels_scaled[:,:4] = labels[:, 1:5]
			labels_scaled[:,:4] = [(xywh2xyxy(torch.tensor(x).view(1, 4)) * gn).view(-1).tolist() for x in labels[:, 1:5]]
			pred_tensor = torch.from_numpy(labels_scaled).to(torch.device(device))
			return pred_tensor
		else:
			return pred
  	
	def read_generated_predictions_all(gtt, saved_path, device):
		# GT_files = [ os.path.splitext(p)[0] for p in os.listdir(gtt) if p.endswith('txt') ]
		# GT_files =[os.path.basename(x).split('.')[0] for x in open(gtt,'r').readlines()  if not x=='' ]		
		GT = dict()
		GT['files'],GT['dirs'] = [], []
		for root, dirs, files in os.walk(gtt, topdown=False):
			for name in sorted(files):
				if name.endswith('txt'):
					GT['files'].append(name.split('.')[0])
					GT['dirs'].append(root)
    
		img = cv2.imread(os.path.join(GT['dirs'][0],GT['files'][0]+'.jpg'))
		hh,ww,_ = img.shape
		shape = hh,ww
		gn = torch.tensor(shape)[[1, 0, 1, 0]]
		
		labels_n, labels, labels_tensor = dict(), dict(), dict()
		count=dict()
		count["gt"], count["pd"], count["files"], count["empty_gt"], count["empty_pd"] = 0,0,0,0,0

		des = ('%22s' + '%15s' * 4) % ('Total Files','Empty GT','Empty PD','GT Boxes','PD Boxes')
		pbar = tqdm(GT['files'], desc=des
              , bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
		for ii, fname in enumerate(pbar):
			fdir = GT['dirs'][ii]
			labels_n[fname], labels[fname], labels_tensor[fname] = dict(), dict(), dict()

			# Join Directory and filename
			f_gt = os.path.join(fdir,fname+'.txt')
			f_pd = os.path.join(saved_path,fname+'.txt')

			# Read text file (GT and PD) and split the columns
			labels_gt = np.array([x.strip(' \n').split(' ') for x in open(f_gt,'r').readlines()  if not x=='' ],dtype=np.float32)
			if not os.path.isfile(f_pd):
				f = open(f_pd,'w')
				f.close()
			labels_pd = np.array([x.strip(' \n').split(' ') for x in open(f_pd,'r').readlines()  if not x=='' ],dtype=np.float32)
			
			# Adjust the columns index according to algo
			if len(labels_pd): 
				if labels_pd.shape[1]==5: labels_pd = labels_pd[:,[1,2,3,4,0]]
				if labels_pd.shape[1]==6: labels_pd = labels_pd[:,[1,2,3,4,5,0]]
			labels_n[fname]['gt'], labels_n[fname]['pd'] = labels_gt, labels_pd
   
			# Change from normalized to image size
			if len(labels_gt): labels_gt[:,1:5] = [(xywh2xyxy(torch.tensor(x).view(1, 4)) * gn).view(-1).tolist() for x in labels_gt[:,1:5]]
			if len(labels_pd): labels_pd[:,0:4] = [(xywh2xyxy(torch.tensor(x).view(1, 4)) * gn).view(-1).tolist() for x in labels_pd[:,0:4]]
			labels[fname]['gt'], labels[fname]['pd'] = labels_gt, labels_pd
			
			# Convert into tensor format
			labels_tensor[fname]['gt'] = torch.from_numpy(labels_gt).to(torch.device(device))
			labels_tensor[fname]['pd'] = torch.from_numpy(labels_pd).to(torch.device(device))
			
			# Update counts
			count['gt'] += len(labels_gt)
			count['pd'] += len(labels_pd)
			if not len(labels_gt): count['empty_gt'] +=1 
			if not len(labels_pd): count['empty_pd'] +=1 
			count['files'] +=1


		des = ('%22s' + '%15s' * 4) % (count['files'],count['empty_gt'],count['empty_pd'],count['gt'],count['pd'])
		print(des)
		print(' ')
  
		# print(f"\t\t\tTotal files:{count['files']}")
		# print(f"\t\t\tEmpty GT files:{count['empty_gt']},\t Empty PD Files:{count['empty_pd']}")
		# print(f"\t\t\tTotal GT Boxes:{count['gt']},\t Total PD Boxes:{count['pd']}\n ")
		return labels_tensor, labels
  



	def read_pesudos(gtt, saved_path, device):
		# GT_files = [ os.path.splitext(p)[0] for p in os.listdir(gtt) if p.endswith('txt') ]
		# GT_files =[os.path.basename(x).split('.')[0] for x in open(gtt,'r').readlines()  if not x=='' ]		
		GT = dict()
		GT['files'],GT['dirs'] = [], []
		for root, dirs, files in os.walk(gtt, topdown=False):
			for name in sorted(files):
				if name.endswith('txt'):
					GT['files'].append(name.split('.')[0])
					GT['dirs'].append(root)
    
		img = cv2.imread(os.path.join(GT['dirs'][0],GT['files'][0]+'.jpg'))
		ww,hh,_ = img.shape
		shape = hh,ww
		gn = torch.tensor(shape)[[1, 0, 1, 0]]
		
		labels_n, labels, labels_tensor = dict(), dict(), dict()
		count=dict()
		count["gt"], count["pd"], count["files"], count["empty_gt"], count["empty_pd"] = 0,0,0,0,0

		des = ('%22s' + '%15s' * 4) % ('Total Files','Empty GT','Empty PD','GT Boxes','PD Boxes')
		pbar = tqdm(GT['files'], desc=des
              , bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
		for ii, fname in enumerate(pbar):
			fdir = GT['dirs'][ii]
			labels_n[fname], labels[fname], labels_tensor[fname] = dict(), dict(), dict()
   
			f_gt = os.path.join(fdir,fname+'.txt')
			f_pd = os.path.join(saved_path,fname+'.txt')
				
			labels_gt = np.array([x.strip(' \n').split(' ') for x in open(f_gt,'r').readlines()  if not x=='' ],dtype=np.float32)
			labels_pd = np.array([x.strip(' \n').split(' ') for x in open(f_pd,'r').readlines()  if not x=='' ],dtype=np.float32)
			if len(labels_pd): labels_pd = labels_pd[:,[1,2,3,4,5,0]]
			labels_n[fname]['gt'], labels_n[fname]['pd'] = labels_gt, labels_pd
   
			if len(labels_gt): labels_gt[:,1:5] = [(xywh2xyxy(torch.tensor(x).view(1, 4)) * gn).view(-1).tolist() for x in labels_gt[:,1:5]]
			if len(labels_pd): labels_pd[:,0:4] = [(xywh2xyxy(torch.tensor(x).view(1, 4)) * gn).view(-1).tolist() for x in labels_pd[:,0:4]]
			labels[fname]['gt'], labels[fname]['pd'] = labels_gt, labels_pd
			
			labels_tensor[fname]['gt'] = torch.from_numpy(labels_gt).to(torch.device(device))
			labels_tensor[fname]['pd'] = torch.from_numpy(labels_pd).to(torch.device(device))
			
			count['gt'] += len(labels_gt)
			count['pd'] += len(labels_pd)
			if not len(labels_gt): count['empty_gt'] +=1 
			if not len(labels_pd): count['empty_pd'] +=1 
			count['files'] +=1


		des = ('%22s' + '%15s' * 4) % (count['files'],count['empty_gt'],count['empty_pd'],count['gt'],count['pd'])
		print(des)
		print(' ')
  
		# print(f"\t\t\tTotal files:{count['files']}")
		# print(f"\t\t\tEmpty GT files:{count['empty_gt']},\t Empty PD Files:{count['empty_pd']}")
		# print(f"\t\t\tTotal GT Boxes:{count['gt']},\t Total PD Boxes:{count['pd']}\n ")
		return labels_tensor, labels
  





	def match_tracks(self,
     				lc_boxes, 		lc_boxes_n,
                  	trk_missed, 	trk_missed_n,
     				trks_confirm, 	trks_confirm_n,
                	trk_new, 		trk_new_n,
					im0,
                	debug = False):
      
		# annotator = self.annotator
  
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
	
			lc_boxes[:,:4] = [self.deepsort._xywh_to_xyxy(box) for box in lc_boxes[:,:4]]
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
					# annotate_boxes(annotator, matched_det_yolo_lc, names, istensor=torch.is_tensor(matched_det_yolo_lc), color=(0,125,255), pad=2) 
					deepsort, count = update_missed_box(deepsort,matched_trk_missed, matched_trk_missed_n, \
																	matched_det_yolo_lc, matched_det_yolo_lc_n, im0, count)
					print('Updated track')
					# if save_txt:
					# 	save_results(matched_trk_missed,matched_trk_missed_n,txt_path,im0) # Save matched missed-lc boxes
					# 	count_saves = count_saves + len(matched_trk_missed)
					# play = False
						  
  
  
  
	def evaluate(self,
				):


		print('Ok')
		return 0









