import numpy as np
import bbox 
from tqdm import tqdm
import os

forward_file = '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_forward'
backward_file = '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_backward'
merged_file = '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_for_back_merged'

if not os.path.exists(merged_file): os.makedirs(merged_file)

forw = sorted({ os.path.splitext(p)[0] : os.path.join(forward_file ,p) for p in os.listdir(forward_file )})
back = sorted({ os.path.splitext(p)[0] : os.path.join(backward_file,p) for p in os.listdir(backward_file)})

for fw,bw in zip(forw,back):

	f_out = open(merged_file+'/'+fw+'.txt','w')
	uniq = []

	f_label = sorted(open(os.path.join(forward_file ,fw+'.txt')).read().split(' \n'))
	b_label = sorted(open(os.path.join(backward_file,bw+'.txt')).read().split(' \n'))


	f_label = [ f for f in f_label if f != '']
	b_label = [ b for b in b_label if b != '']



	for fbox in f_label:
		if fbox in b_label:
			pass
		else:
			uniq.append(fbox)
			print(f'{fbox} is not found in b_label')
   
	for bbox in b_label:
		data = str(bbox)
		f_out.write(data+'\n')
  
	if len(uniq):	
		for bbox in uniq:
			data = str(bbox)
			f_out.write(data+'\n')
   

	if not len(f_label) == len(b_label):
		# print(f'Check.{len(f_label)} != {len(b_label)}')
		pass