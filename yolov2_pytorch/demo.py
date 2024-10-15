import os
import shutil
import argparse
import time
import torch
from torch.autograd import Variable
# from PIL import Image
# from test import prepare_im_data
# from yolov2 import Yolov2
from yolov2_pytorch.config import config as cfg
from yolov2_pytorch.yolov2_tiny_2 import Yolov2
from yolov2_pytorch.yolo_eval import yolo_eval
# from util.visualize import draw_detection_boxes
# import matplotlib.pyplot as plt
from yolov2_pytorch.util.network import WeightLoader
import cv2
import numpy as np
import yolov2_pytorch.TTA as TTA


def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- cv.image

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """
    
    im_info = dict()
    # im_info['width'], im_info['height'] = img.size
    im_info['height'], im_info['width'] , _ = img.shape

    # resize the image
    H, W = cfg.input_size
    # im_data = im_data.resize((H, W))
    img = cv2.resize(img, (W,H))
    im_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # to torch tensor
    # im_data = torch.from_numpy(np.array(im_data)).float() / 255
    im_data = torch.from_numpy(im_data).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)         # specify the path of output label folder here in default
    parser.add_argument('--model_name', dest='model_name',
                        default='data/pretrained/yolov2-tiny-voc.pth', type=str) # specift the path of weights here in default
    parser.add_argument('--use_cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--data', type=str,
                        default=None,
                        help='Path to data.txt file containing images list') # specify the data file (name should be data.txt) which should be a text file containing list of paths of all the inference images

    parser.add_argument('--classes', type=str,
                        default=None,
                        help='provide the list of class names if other than voc') # Example 'default = ('Vehicle', 'Rider', 'Pedestrian')'
    parser.add_argument('--conf_thres', type=float,
                        default=0.001,
                        help='choose a confidence threshold for inference, must be in range 0-1')
    parser.add_argument('--nms_thres', type=float,
                        default=0.45,
                        help='choose an nms threshold for post processing predictions, must be in range 0-1s')
    parser.add_argument('--pseudo_type', type=str,
                        default='low_conf',
                        help='Type of pseudo-label generation can be either high_conf, low_conf or None')
    args = parser.parse_args()
    return args

def demo(args):
    print('call with args: {}'.format(args))
    
    if args.pseudo_type=='low_conf':
        thres = 0.1
    elif args.pseudo_type == 'high_conf':
        thres = 0.45
    elif args.pseudo_type is None:
        print('regular inference...')        
    
    if args.data==None:
        # input images
        images_dir   = 'images'
        images_names = ['image1.jpg', 'image2.jpg']
    else:
        with open(args.data, 'r') as f:
            images_names = f.readlines()
    
    try: args.classes
    except: args.classes=None
    
    if args.classes is None:
        classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')
    else:
        classes = args.classes        
    
    # set the save folder path for predictions
    if args.data==None:
        pred_dir = os.path.join(images_dir, 'preds')
        pred_dir = os.path.join(os.getcwd(),pred_dir)    
    else:
        # dir      = args.data.split('/data')[0]
        # pred_dir = os.path.join(dir, 'preds')
        pred_dir = args.output_dir
#        pred_dir = os.path.join(os.getcwd(),pred_dir)                    
    
    if not os.path.exists(pred_dir):
        print(f'making {pred_dir}')
        os.makedirs(pred_dir)
    else:
        print('Deleting existing pred dir')
        shutil.rmtree(pred_dir, ignore_errors=True)
        print(f'making new {pred_dir}')
        os.makedirs(pred_dir)
    
    try: args.weights
    except: args.weights=None
    
    if args.weights is None:
        model = Yolov2(classes=classes)
        
        model_type = args.model_name.split('.')[-1]
        
        if model_type == 'weights':
            weight_loader = WeightLoader()
            weight_loader.load(model, 'yolo-voc.weights')
            print('loaded')

        else:
            model_path = args.model_name
            print('loading model from {}'.format(model_path))
            if args.use_cuda:
                checkpoint = torch.load(model_path, map_location=('cuda:0'))
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
    else:
        model = Yolov2(classes=classes)
        checkpoint = args.weights
        model.load_state_dict(checkpoint['model'])


    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')

    for image_name in images_names:
        if args.data==None:
            image_path = os.path.join(images_dir, image_name)
            # img        = Image.open(image_path)
            img = cv2.imread(image_path, 3)
        else:
            image_path = image_name.split('\n')[0]
            # img        = Image.open(image_path)
            img = cv2.imread(image_path, 3)   
        
        if args.pseudo_type == 'high_conf':     # predictions with TTA for high conf-pseudo-labels
            TTA_Dict     =  TTA.TRANSFORM(img)
            _merged_pred =  torch.zeros((6))
            for key, _img in TTA_Dict.items():
                im_data, im_info = prepare_im_data(_img)

                if args.use_cuda:
                    im_data_variable = Variable(im_data).cuda()
                else:
                    im_data_variable = Variable(im_data)

                tic = time.time()

                yolo_output = model(im_data_variable)
                yolo_output = [item[0].data for item in yolo_output]
                _detections  = yolo_eval(yolo_output, im_info, conf_threshold=args.conf_thres, nms_threshold=args.nms_thres)
                if len(_detections)>0:
                    det_boxes   = _detections[:, :5] #.cpu().numpy()
                    det_classes = _detections[:, -1] #.long().cpu().numpy()
                    # pred = np.zeros((det_boxes.shape[0],6))
                    pred = torch.zeros((det_boxes.shape[0],6))
                    pred[:, :5] = det_boxes
                    pred[:,-1]  = det_classes
                    pred = TTA.UNDO(pred,img_info=(1280,720), mode = key)
                    _merged_pred = torch.vstack((_merged_pred, pred)) if isinstance(pred, torch.Tensor) else np.vstack((_merged_pred, pred))
            merged_pred  = TTA.delete(_merged_pred, 0,0) if isinstance(pred, torch.Tensor) else np.delete(_merged_pred, 0,0) # deleting first row of zeros
            try: merged_pred
            except: merged_pred=None
            if merged_pred.shape is not None:
                nms_keep = TTA.NMS(merged_pred, iou_threshold=0.2)
                nms_keep =  TTA.filter_boxes(merged_pred, nms_keep)
                detections, _ = TTA.CLEAN_BOXES(nms_keep)                
            # print()
        elif args.pseudo_type == 'low_conf':        # predictions without TTA for low-conf pseudo-labels
            im_data, im_info = prepare_im_data(img)

            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            tic = time.time()

            yolo_output = model(im_data_variable)
            yolo_output = [item[0].data for item in yolo_output]
            detections  = yolo_eval(yolo_output, im_info, conf_threshold=args.conf_thres, nms_threshold=args.nms_thres)       
        
        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        
        if len(detections)>0:
            name        = image_name.split('/')[-1]
            pred_name   = name.split('.')[0] + '.txt'
            pred_path   = os.path.join(pred_dir, pred_name)
            det_boxes   = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            pred        = np.zeros((det_boxes.shape[0],6))
            pred[:, :5] = det_boxes
            pred[:,-1]  = det_classes # pred is [x, y, w, h, conf, cls]
            
            # save predictions if generation pseudo-labels
            preds = [] 
            if args.pseudo_type is not None:
                for i in range(pred.shape[0]):
                    _pred = pred[i]
                    if (_pred[4] >= thres):
                        label_line = str(int(_pred[-1])) + ' {} {} {} {} {} \n'.format(round(_pred[0]/im_info['width'], 7),
                                                                round(_pred[1]/im_info['height'], 7),
                                                                round(_pred[2]/im_info['width'], 7),
                                                                round(_pred[3]/im_info['height'], 7),
                                                                round(_pred[4], 7)) # line formate is (category, x, y, width, height, conf)
                        preds.append(label_line)
                # write the predictions of current image to the write location
                f = open(pred_path, 'w')
                f.writelines(preds)            
            print(f'prediction result for {image_path} is saved...')
    print('done...')    

if __name__ == '__main__':
    args = parse_args()
    demo(args)