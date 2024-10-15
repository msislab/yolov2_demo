import numpy as np
import cv2
import os
from tqdm import tqdm

play = True 


# DIRNAME_PREDICTION = "/data/pseudos/Algo_BDD_1Seq-0.5"
# # DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/yolov5/runs/val/mAP_test_BDD_b1d0a191-2ed2269e_conf_50/labels"
# # # DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_b1d0a191_normalized_deepsort"
# # # DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_b1d0a191_normalized_w_initialized"
# # # DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_b1d0a191_normalized_w_recovered_forward"
# # # DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_b1d0a191_normalized_w_recovered_backward"
# # DIRNAME_RECOVERED = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_b1d0a191_normalized_w_recovered_for_back_merged"
# DIRNAME_RECOVERED   = "/data/pseudos/Algo_BDD_1Seq-0.5-Merged"
# FILENAME_GROUNDTRUTH = "/data/codes/30_yolo/yolov5/1seq.txt"

# DIRNAME_PREDICTION = "/data/pseudos/Algo_BDD_1Seq-0.5"
# DIRNAME_RECOVERED   = "/data/pseudos/Algo_BDD_1Seq-0.5-Merged"
# FILENAME_GROUNDTRUTH = "/data/codes/30_yolo/yolov5/1seq.txt"


# DIRNAME_PREDICTION = "/data/pseudos/Algo_BDD_200Seq_40k-0.5"
# DIRNAME_RECOVERED   = "/data/pseudos/Algo_BDD_200Seq_40k-0.5-Merged"
# FILENAME_GROUNDTRUTH = "/data/datasets/4_BDD/1_BDD_Complete/val_list.txt"

# DIRNAME_PREDICTION = "/data/pseudos/CJ-0.01"
DIRNAME_PREDICTION = "/data/pseudos/CJ-0.5"
# # DIRNAME_RECOVERED = "/data/pseudos/CJ-0.5-FW"
# # DIRNAME_RECOVERED = "/data/pseudos/CJ-0.5-BW"
DIRNAME_RECOVERED = "/data/pseudos/CJ-0.5-FW"
FILENAME_GROUNDTRUTH = "/data/datasets/8_Cheongju/CJ_bdd/val_list.txt"




# Full Val
# DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/yolov5/runs/val/mAP_Pseudo_labels_BDD_Full_Val_conf50/labels"
# DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort"
# DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_modified"
# DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_forward"
# DIRNAME_PREDICTION = "/data/codes/30_yolo/test/Yolov5_DeepSORT_PseudoLabels/output/BDD_VAL_ALL_deepsort_backward"
# FILENAME_GROUNDTRUTH = "/data/datasets/4_BDD/1_BDD_Complete/val_list.txt"

print(f"Comparing labels for \n\t{DIRNAME_PREDICTION}\n\n\t{DIRNAME_RECOVERED}\n\n\t{FILENAME_GROUNDTRUTH}\n\n")
DICT_TEXTNAMES_PREDICTION = { os.path.splitext(p)[0] : os.path.join(DIRNAME_PREDICTION,p) for p in os.listdir(DIRNAME_PREDICTION)}
DICT_TEXTNAMES_RECOVERED = { os.path.splitext(p)[0] : os.path.join(DIRNAME_RECOVERED,p) for p in os.listdir(DIRNAME_RECOVERED)}

with open(FILENAME_GROUNDTRUTH) as f:
	IMAGENAMES_GROUNDTRUTH = sorted(f.read().splitlines())

namelist = ["car","bus","truck","motorcycle","bicycle","trailer","train","rider","pedestrian","traffic light","traffic sign","other vehicle","other person"]

with open("/data/datasets/0_Scripts/bdd.names") as f:
	NAMES_CLASS = f.read().splitlines()
NUMBER_CLASSES = len(NAMES_CLASS)

###
print("# of data: %d"%len(IMAGENAMES_GROUNDTRUTH))

count_rc_total = 0
for index in tqdm(range(len(IMAGENAMES_GROUNDTRUTH))):
	
	count_gt, count_pd, count_rc = 0, 0, 0
 
	imagename = IMAGENAMES_GROUNDTRUTH[index]
	try:
		textname_prediction = DICT_TEXTNAMES_PREDICTION[ os.path.splitext( os.path.basename(imagename) )[0] ]
	except:
		f = open(os.path.join( DIRNAME_PREDICTION , os.path.basename(imagename)[:-4]+'.txt' ) , 'a')
		f.close()
	try:
		textname_recovered = DICT_TEXTNAMES_RECOVERED[ os.path.splitext( os.path.basename(imagename) )[0] ]
	except:
		f1 = open(os.path.join( DIRNAME_RECOVERED , os.path.basename(imagename)[:-4]+'.txt' ) , 'a')
		f1.close()
  
	textname_prediction = textname_prediction.replace("jpg","txt")
	textname_recovered = textname_recovered.replace("jpg","txt")
	textname_groundtruth = imagename.replace("jpg","txt")

	with open(textname_groundtruth) as f:
		info_groundtruth = f.read().splitlines()
	bboxes_groundtruth = []
	labels_groundtruth = []
	for bbox in info_groundtruth:
		bbox = bbox.split()
		label = int(bbox[0])
		#label = 0
		bboxes_groundtruth.append([float(c) for c in bbox[1:5]])
		labels_groundtruth.append(label)
		count_gt+=1

	with open(textname_prediction) as f:
		info_prediction = f.read().splitlines()
	bboxes_prediction = []
	labels_prediction = []
	scores_prediction = []
	for bbox in info_prediction:
		bbox = bbox.split()
		label      = int(bbox[0])
		confidence = 0.8
		if label  in [0,1,2,3,4,5,6,7,8]:
			bboxes_prediction.append([float(c) for c in bbox[1:5]])
			labels_prediction.append(label)
			scores_prediction.append(confidence)
			count_pd+=1
  
  
	with open(textname_recovered) as f:
		info_recovered = f.read().splitlines()
	bboxes_recovered = []
	labels_recovered = []
	scores_recovered = []
	for bbox in info_recovered:
		bbox = bbox.split()
		label      = int(bbox[0])
		confidence = 0.8
		if label  in [0,1,2,3,4,5,6,7,8]:
			bboxes_recovered.append([float(c) for c in bbox[1:5]])
			labels_recovered.append(label)
			scores_recovered.append(confidence)
			count_rc+=1
 
 
 

	if not count_rc==count_pd:
	# if True:
		count_rc_total+= (count_rc - count_pd)
	
		im0 = cv2.imread(imagename)
	

		scale_percent = 50 # percent of original size
		font_siz = (scale_percent/100)*3 # percent of original size
		cv2.putText(im0  , str(os.path.basename(imagename)),(50, 40), cv2.FONT_HERSHEY_SIMPLEX, font_siz/1.7, (0,0,0), 2, cv2.LINE_AA)
		cv2.putText(im0  , "--- Ground Truth   ", 			(50,100), cv2.FONT_HERSHEY_SIMPLEX, font_siz/2, (0,0,255), 2, cv2.LINE_AA)
		cv2.putText(im0  , "--- Predicted Boxes", 			(50,140), cv2.FONT_HERSHEY_SIMPLEX, font_siz/2, (255,0,255), 2, cv2.LINE_AA)
		cv2.putText(im0  , "--- Recovered Boxes", 			(50,180), cv2.FONT_HERSHEY_SIMPLEX, font_siz/2, (0,255,255), 2, cv2.LINE_AA)
		# cv2.putText(im0  , f"--- Recovered Boxes ({count_rc - count_pd}) (T: {count_rc_total})", 			(50,180), cv2.FONT_HERSHEY_SIMPLEX, font_siz/2, (0,255,255), 2, cv2.LINE_AA)
		# cv2.putText(im1  , str(fw) ,				   		(50, 40), cv2.FONT_HERSHEY_SIMPLEX, font_siz/1.7, (0,0,0), 2, cv2.LINE_AA)
		# cv2.putText(im1  , "--- Groun

 
		print(f'uneven, gt:{count_gt}, pred:{count_pd}, reco:{count_rc}')
		play = False
  
		img_h,img_w,_ = im0.shape
  
		for bbox_class, [bbox_x, bbox_y ,bbox_w, bbox_h] in zip(labels_groundtruth,bboxes_groundtruth):
			# if bbox_class in [0,1,2,3,4,5,6,7,8]:
				start = (int((bbox_x-(bbox_w/2))*img_w), int((bbox_y-(bbox_h/2))*img_h))
				end = (int((bbox_x+(bbox_w/2))*img_w), int((bbox_y+(bbox_h/2))*img_h))
				class_box = (start[0]+len(namelist[bbox_class]*5), start[1]-11)
				box_color = (0,0,255)
				cv2.rectangle(im0, start, end, box_color, thickness=2)
				cv2.rectangle(im0, class_box, start, box_color, thickness=-1)
				cv2.rectangle(im0, class_box, start, box_color, thickness=2)
				cv2.putText(  im0, namelist[bbox_class], (start[0], start[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
		
    
		for bbox_class, [bbox_x, bbox_y ,bbox_w, bbox_h] in zip(labels_recovered,bboxes_recovered):
			# if bbox_class in [0,1,2,3,4,5,6,7,8]:
				start = (int((bbox_x-(bbox_w/2))*img_w), int((bbox_y-(bbox_h/2))*img_h))
				end = (int((bbox_x+(bbox_w/2))*img_w), int((bbox_y+(bbox_h/2))*img_h))
				class_box = (start[0]+len(namelist[bbox_class]*5), start[1]-11)
				box_color = (0,255,255)
				cv2.rectangle(im0, start, end, box_color, thickness=2)
				cv2.rectangle(im0, class_box, start, box_color, thickness=-1)
				cv2.rectangle(im0, class_box, start, box_color, thickness=2)
				cv2.putText(  im0, namelist[bbox_class], (start[0], start[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    
			
  
		for bbox_class, [bbox_x, bbox_y ,bbox_w, bbox_h] in zip(labels_prediction,bboxes_prediction):
			# if bbox_class in [0,1,2,3,4,5,6,7,8]:
				start = (int((bbox_x-(bbox_w/2))*img_w), int((bbox_y-(bbox_h/2))*img_h))
				end = (int((bbox_x+(bbox_w/2))*img_w), int((bbox_y+(bbox_h/2))*img_h))
				class_box = (start[0]+len(namelist[bbox_class]*5), start[1]-11)
				box_color = (255,0,255)
				cv2.rectangle(im0, start, end, box_color, thickness=2)
				cv2.rectangle(im0, class_box, start, box_color, thickness=-1)
				cv2.rectangle(im0, class_box, start, box_color, thickness=2)
				cv2.putText(  im0, namelist[bbox_class], (start[0], start[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    
		
  
		new_width = 1000
		dim = (int(new_width * (im0.shape[1]/im0.shape[0])), int(new_width))
		
		# resized_gt = cv2.resize(im1, dim, interpolation = cv2.INTER_AREA)
		# cv2.imshow("GT", resized_gt)
		
		resized_predictions = cv2.resize(im0, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow("Pred", resized_predictions)
		cv2.imwrite("test.jpg",im0)
		inkey = cv2.waitKey(100)

		if play: 
			inkey = cv2.waitKey(1)
		else: 
			inkey = cv2.waitKey(0)
			
		if inkey == ord("q"):
			break
		if inkey == ord(" "):
			play = not play
		if inkey == ord("f"):
			play = False
		if inkey == ord("b"):
			# dataset.rev = True
			play = False