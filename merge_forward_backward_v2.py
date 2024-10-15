import os
from pathlib import Path
import argparse

def merge(opt):

	forward_file = opt.forward # '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_forward'
	backward_file = opt.backward # '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_backward'
	merged_file = opt.merged # '/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_for_back_merged'

	Path(merged_file).mkdir(parents=True, exist_ok=True)

	forw = sorted({ os.path.splitext(p)[0] : os.path.join(forward_file ,p) for p in os.listdir(forward_file )})
	back = sorted({ os.path.splitext(p)[0] : os.path.join(backward_file,p) for p in os.listdir(backward_file)})

	count=dict()
	count['f'],count['b'],count['m'] = 0,0,0
	for fw in forw:

		f_out = open(merged_file+'/'+fw+'.txt','w')
		uniq = []

		f_file = os.path.join(forward_file ,fw+'.txt')
		b_file = os.path.join(backward_file,fw+'.txt')

		# if not os.path.isfile(f_file): 
		# 	ffile = open(f_file,'w')
		# 	ffile.close()

		# if not os.path.isfile(b_file): 
		# 	bfile = open(b_file,'w')
		# 	bfile.close()


		try:
			f_label = sorted(open(f_file).read().split(' \n'))
		except:
			tmp = open(f_file,'w')
			tmp.close()
			f_label = sorted(open(f_file).read().split(' \n'))
		
		try:
			b_label = sorted(open(b_file).read().split(' \n'))
		except:
			tmp = open(b_file,'w')
			tmp.close()
			b_label = sorted(open(b_file).read().split(' \n'))
			


		f_label = [ f for f in f_label if f != '']
		b_label = [ b for b in b_label if b != '']

		for fbox in f_label:
			if fbox in b_label:
				pass
			else:
				uniq.append(fbox)
				# print(f'{fbox} is not found in b_label')
	
		for bbox in b_label:
			data = str(bbox)
			f_out.write(data+'\n')
	
		if len(uniq):	
			for bbox in uniq:
				data = str(bbox)
				f_out.write(data+'\n')
	
		count['f'], count['b'], count['m']  = count['f'] + len(f_label)   ,    count['b'] + len(b_label)   ,    count['m'] + len(uniq) + len(b_label)   
	
	print(f"\n\tMerged labels saved in {opt.merged}\n")
	print(f"\tLabels in Forward Sequence  {count['f']}     ({count['m']-count['f']} less)")
	print(f"\tLabels in Backward Sequence {count['b']}     ({count['m']-count['b']} less)")
	print(f"\tLabels in Merged files      {count['m']}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--forward', type=str, default='')
	parser.add_argument('--backward', type=str, default='')
	parser.add_argument('--merged', type=str, default='')
	# opt = parser.parse_args()
	opt, unknown = parser.parse_known_args()

	merge(opt)
