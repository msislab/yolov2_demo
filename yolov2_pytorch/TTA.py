import torch
# from torch.autograd import Variable
# from PIL import Image, ImageOps
import cv2
# from yolov2 import Yolov2
# import torchvision.transforms as transforms
import numpy as np


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)

# def REVERT_prepare_im_data(im_data, im_info):
#     """
#     Prepare image data that will be feed to TTA.
#     Arguments:  Img -- Tensor
#                 Original image info
#     Returns:    Array image.
#     """
#     # width, height = im_info['width'], im_info['height']
#     width, height = int(im_info[0]), int(im_info[1])
#     im_data = im_data.squeeze()
#     # im_data = im_data.permute(2,1,0) # (3,640,640) --> (640,640,3)
#     # im_data = im_data*255
    
#     trans =  transforms.ToPILImage()
#     img   =  trans(im_data)
#     img   =  img.resize((width,height))

#     return img

def TRANSFORM(img):
    """
    This function is used as transformation for test with augmentation
    Input:   Images
    Returns: Transformed Image 
    
    """
    TTA_Dic = {}

    # cv2.imshow('', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # image_path = img.split('\n')[0]
    # img   =  Image.open(image_path)
    TTA_Dic["Original"] = img
    
    h, w = img.shape[:2]
    # img2  =  ImageOps.mirror(img)
    img2  =  cv2.flip(img,1)
    TTA_Dic['Flipped'] = img2

    # cv2.imshow('', img2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # sf_x, sf_y = 2, 2
    img3 = cv2.resize(img, (w*2,h*2), interpolation = cv2.INTER_LINEAR)
    # img3  =  img.resize((int(img.size[0]*sf_x), int(img.size[1]*sf_y)),Image.LANCZOS)    
    TTA_Dic["Upscaled"] = img3
    
    img4  =  cv2.flip(img3,1)
    TTA_Dic["Both"] = img4
    
    return TTA_Dic

def UNDO(BB, img_info, mode = ""):
    '''
    This function changes the bounding box coordinates from transformed
    image to original image
    
    Input: Tesnor or NumPy array (N,6) (x1, y1, x2, y2,  conf, class)
    Mode: Flip, Upscale or both
    
    Returns: Bounding box coodinates w.r.t original image. 
    
    '''
    
    img_width, img_height = img_info
    
        
    if mode == "Flipped":
        unflipped_boxes = BB.clone() if isinstance(BB, torch.Tensor) else np.copy(BB) # clone if the input is tensor, copy if the input is numpy
        for i in range (len(BB)):
            unflipped_boxes[i,0:1] = img_width-BB[i,2:3]-1   # (width - x - 1)  ---->  
            unflipped_boxes[i,2:3] = img_width-BB[i,0:1]-1    
        return unflipped_boxes
    
    elif mode =="Upscaled": 
        # original_box = np.copy(BB) # copying all the values of the upscaled bounding box
        original_box = BB.clone() if isinstance(BB, torch.Tensor) else np.copy(BB)
        original_box[:,0:4]/=2     # downscaling the coordinates by dividing x1y1x2y2 by 2
        return original_box

    elif mode =="Both": # returning original box
        # scale_box = np.copy(BB)
        scale_box = BB.clone() if isinstance(BB, torch.Tensor) else np.copy(BB)
        scale_box[:,0:4]/=2
        
        original_box = scale_box.clone() if isinstance(scale_box, torch.Tensor) else np.copy(scale_box)
        for i in range (len(scale_box)):
            original_box[i,0:1] = img_width-scale_box[i,2:3]-1   # (width - x - 1)  ---->  
            original_box[i,2:3] = img_width-scale_box[i,0:1]-1    
        return original_box
        
    else:     # returning original box
        return BB
        
def IOU(boxes_a, boxes_b):

    def box_area(box):return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    if isinstance((boxes_a), torch.Tensor):
        top_left     =  torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
        bottom_right =  torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])
        area_inter   =  (bottom_right - top_left).clamp(0).prod(2)      
        return area_inter / (area_a[:, None] + area_b - area_inter) 
    
    else:
        top_left     =  np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
        bottom_right =  np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])
        #intersection(x,y) = (bottom_right(x,y,2) - Top_left(x,y,2)), climp, prod
        area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        return area_inter / (area_a[:, None] + area_b - area_inter)       

def NMS(predictions, iou_threshold = 0.5):
    
    rows, _ = predictions.shape

    sort_index  = torch.flip((predictions[:,4].argsort()),[0]) if isinstance(predictions, torch.Tensor) else np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes   =  predictions[:, :4]
    classes =  predictions[:, 5]
    ious    =  IOU(boxes, boxes)
    ious    =  ious - torch.eye(rows) if isinstance(predictions, torch.Tensor) else ious - np.eye(rows) 
    keep    =  torch.ones(rows, dtype=bool) if isinstance(predictions, torch.Tensor) else np.ones(rows, dtype=bool)

    for index, (iou, _class) in enumerate(zip(ious, classes)):
        if not keep[index]:continue

        condition = (iou > iou_threshold) & (classes == _class)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

def filter_boxes(boxes, results):
    
    _, column = boxes.shape
    keep    = torch.zeros((column)) if isinstance(boxes, torch.Tensor) else np.zeros((column))  # initializing an empty array/tensor
    for i in range(len(results)):
        if results[i]:
            keep = torch.vstack((keep,boxes[i,:])) if isinstance(boxes, torch.Tensor) else np.vstack((keep,boxes[i,:]))
    keep = delete(keep,0,0) if isinstance(boxes, torch.Tensor) else np.delete(keep, 0, 0) # deleting the first row of zeros in array/tensor
    return keep


 
def CLEAN_BOXES(Boxes, threshold = 153600):
    ''' 
    This function removes objects from label file with respect to the given threshold. 
    Input:   Path:Image path
             Threshold: Area 

    Returns: Labels without certain objects, labels_remaining. 
    '''
    _, column = Boxes.shape
    def CHECK_BOX(box, threshold ):

        Big_box_detected = False
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > threshold:
            Big_box_detected = True
        
        return Big_box_detected
    
    no_labels_removed  =  0 
    cleaned_boxes      =  torch.zeros((column)) if isinstance(Boxes, torch.Tensor) else np.zeros((column))
    
    for i in range(len(Boxes)):        
        box_status = CHECK_BOX(Boxes[i,:],threshold)
        if not box_status:
            cleaned_boxes = torch.vstack((cleaned_boxes,Boxes[i,:])) if isinstance(Boxes, torch.Tensor) else np.vstack((cleaned_boxes,Boxes[i,:]))
        else:no_labels_removed+=1 
    
    cleaned_boxes = delete(cleaned_boxes,0,0) if isinstance(Boxes, torch.Tensor) else np.delete(cleaned_boxes,0,0)
    return cleaned_boxes, no_labels_removed