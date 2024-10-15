from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
import yaml
import os
import numpy as np

def read_yaml(path,new_path,save_labels,min_w=0,min_h=0,ar=0):

	with open(path, errors='ignore') as f:
		data = yaml.safe_load(f)  # load hyps dict
	
	print(f"Training Data:",end=' ')
	data_train = read_labels('train',data['train'],new_path,save_labels,min_w,min_h,ar)

	print(f"Validation Data:",end=' ')
	data_val = read_labels('val',data['val'],new_path,save_labels,min_w,min_h,ar)

	ret_data=dict()
	ret_data["data_train"]=data_train
	ret_data["data_val"]=data_val

	return ret_data



def read_labels(phase,path,new_path,save_labels,min_w=0,min_h=0,ar=0,classes=[]):

	if save_labels: Path(os.path.join(new_path,phase)).mkdir(parents=True, exist_ok=True)

	with open(path) as f:
		data = sorted(f.read().splitlines())
	print(f"{path}")

	labels=dict()
	counts=dict()
	clean_labels=dict()

	counts[f'width_less_than_{min_w}'],counts[f'height_less_than_{min_h}'],counts[f'aspect_ratio_larger_than_{ar}'],counts['total_boxes'],counts[f'height_width_less_than_{min_h}_{min_w}'],counts[f'height_width_less_than_{min_h}_{min_w}_ar_larger_than_{ar}']=0,0,0,0,0,0
	pbar = tqdm(data)
	pbar.set_description( f"Box_W < {min_w} = {counts[f'width_less_than_{min_w}']}, Box_H < {min_h} = {counts[f'height_less_than_{min_h}']}, Box_AR > {ar} = {counts[f'aspect_ratio_larger_than_{ar}']}")
	for imagename in pbar:
		# if idx<33290:
		# 	continue
		im = cv2.imread(imagename)
		H,W,C = im.shape
		# H,W,C = 720,1280,3
		textname_groundtruth = imagename.replace("jpg","txt")
		ffile=os.path.basename(imagename).split('.')[0]
		try:
			with open(textname_groundtruth) as f:
				info_groundtruth = f.read().splitlines()
		except:
			continue
		labels[ffile], clean_labels[ffile]=dict(),dict()
		clean_labels[ffile]["nr_boxes"],clean_labels[ffile]["classes"],clean_labels[ffile]["out_line"] = [],[],[]
		labels[ffile]["nr_boxes"],labels[ffile]["boxes"],labels[ffile]["box_details"],labels[ffile]["classes"] = [],[],dict(),[]
		labels[ffile]["image"], labels[ffile]["label"] = imagename, imagename.replace('.jpg', '.txt')
		labels[ffile]["box_details"]['width'],labels[ffile]["box_details"]['height']=[],[]
		for bbox in info_groundtruth:
			bbox = bbox.split()
			label = int(bbox[0])
			if label in classes or classes==[]:
				clean_flag=True
				labels[ffile]["nr_boxes"].append([float(c) for c in bbox[1:5]])
				labels[ffile]["classes"].append(label)
				nr_box = bbox[1:5]
				x,y,w,h = bbox[1:5]
				bbox[1:5] = float(x)*W,float(y)*H,float(w)*W,float(h)*H
				labels[ffile]["boxes"].append([int(c) for c in bbox[1:5]])
				labels[ffile]["box_details"]['width'].append(bbox[3])
				labels[ffile]["box_details"]['height'].append(bbox[4])
				
				counts['total_boxes']+=1
				if (not min_w==0 or not min_h==0 or not ar==0) and not (bbox[3]==0 or bbox[4]==0):
					if ((bbox[3] < min_w) or (bbox[4] < min_w)) or (bbox[3]/bbox[4] > ar or bbox[4]/bbox[3] > ar ): 
						if bbox[3] < min_w and bbox[4] < min_w and (bbox[3]/bbox[4] > ar or bbox[4]/bbox[3] > ar ): 
							counts[f'height_width_less_than_{min_h}_{min_w}_ar_larger_than_{ar}']+=1
						elif bbox[3] < min_w and bbox[4] < min_w: 
							counts[f'height_width_less_than_{min_h}_{min_w}']+=1
						elif bbox[3] < min_w: 
							counts[f'width_less_than_{min_w}']+=1
						elif bbox[4] < min_w: 
							counts[f'height_less_than_{min_h}']+=1
						elif bbox[3]/bbox[4] > ar or bbox[4]/bbox[3] > ar : 
							counts[f'aspect_ratio_larger_than_{ar}']+=1
						else:
							print(f"Please check {imagename}")
						
						clean_flag=False
	
				if clean_flag:
					clean_labels[ffile]["nr_boxes"].append([float(c) for c in nr_box])
					clean_labels[ffile]["classes"].append(label)
					out_string = f"{str(label)} {str(nr_box[0])} {str(nr_box[1])} {str(nr_box[2])} {str(nr_box[3])}\n"
					clean_labels[ffile]["out_line"].append(out_string)
			
		if save_labels:
			# img_path = os.path.join(new_path,ffile+'.jpg')
			# shutil.copyfile(imagename,img_path)

			lab_path = os.path.join(new_path,phase,ffile+'.txt')

			f = open(lab_path, 'a')
			if len(clean_labels[ffile]["out_line"]) > 0:
				for linee in clean_labels[ffile]["out_line"]:
					f.write(linee)  # to YOLO format                        
					# f.write(('%g ' * 5  + '\n') % (cls, l, t, w, h))  # to YOLO format  pass
			pass
			
				
		tbox,lwb,lhb,larb = counts['total_boxes'],counts[f'width_less_than_{min_w}'],counts[f'height_less_than_{min_h}'],counts[f'aspect_ratio_larger_than_{ar}']
		lwhb = counts[f'height_width_less_than_{min_h}_{min_w}']
		lwhbgarb = counts[f'height_width_less_than_{min_h}_{min_w}_ar_larger_than_{ar}']
		pbar.set_description( f"Total = {tbox},  H < {min_h} & W < {min_w} & AR > {ar}= {lwhbgarb} ({round(lwhbgarb/tbox,2)}%),  H < {min_h} & W < {min_w}= {lwhb} ({round(lwhb/tbox,2)}%),  W < {min_w} = {lwb} ({round(lwb/tbox,2)}%),  H < {min_h} = {lhb} ({round(lhb/tbox,2)}%),  AR > {ar} = {larb} ({round(larb/tbox,2)}%)")

	out_data = dict()
	out_data["labels"] = labels
	out_data["counts"] = counts
	out_data["clean_labels"] = clean_labels

	lenboxes=0
	for i in labels.keys():
		lenboxes+=len(labels[i]['classes'])
	print(f"Total boxes in labels: {lenboxes}")

	lenboxes=0
	for i in clean_labels.keys():
		lenboxes+=len(clean_labels[i]['classes'])
	print(f"Total boxes in cleaned labels: {lenboxes}")

	return out_data


