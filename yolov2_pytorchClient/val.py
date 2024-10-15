from tqdm import tqdm
import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from yolov2_tiny import Yolov2
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, Custom_yolo_dataset
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
from config import config as cfg
import pascalvoc
import shutil
import warnings
warnings.filterwarnings('ignore')



def parse_args():


    parser = argparse.ArgumentParser(description='Yolo v2') #Check Description
    
    parser.add_argument("--weights", dest="weights",
                        type = str, help = "Weight file path",
                        default = 'yolov2_bestmap.pth'  
                        ) #Check default path
    
    parser.add_argument("--cfg", dest = "cfg",
                        type = str, help = "Enter Configuration File",
                        default = ""                     
                        ) # Check default file path
    
    parser.add_argument("--nc", dest = "num_classes",
                        type = int, help = "Enter Number of Classes", 
                        default=""
                        ) # Check default number of classes

    parser.add_argument("--nw", dest = "num_workers",
                        type = int, help = "number of workers to load training data", 
                        default = 1 
                        )

    parser.add_argument("--val_data", dest = "val_data", 
                        type = str, help="Enter custom dataset path", 
                        default = " "
                        ) #Check Default path 

    parser.add_argument("--customData", dest = "customData",
                        type = bool, help = "Using Custom data or not", 
                        default = False
                        )

    parser.add_argument("--batch_size", dest = "batch_size",
                        type = int, help = "Enter batch_size",
                        default = " "
                        )#check default batchsize for validation 

    parser.add_argument("--temp_path", dest = "temp_path", 
                        type = str, help = "Enter output directory", 
                        default = " " 
                        )# Check default path

    parser.add_argument("--use_cuda", dest = "use_cuda",
                        type = bool, help = "Use CUDA or not", 
                        default = True
                        )
    
    args = parser.parse_args()

    return args


def appendLists(a=[],b=[]):
    for i in range(len(b)):
        _smal_list = f'{int(b[i][0])} {round(b[i][1], 2)} {round(b[i][2],4)} {round(b[i][3],4)} {round(b[i][4],4)} {round(b[i][5],4)} \n'
        a.append(_smal_list)
    return a 

def test_for_train(temp_path, model, args, val_data=None, _num_classes=0, customData=False, withTrain=False):
    # args = parse_args()
    # make a directory to save predictions paths
    
    save_dir = 'output/preds/labels' #Check Output directory for validation  
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'{save_dir} was existing and removed...')
        os.mkdir(save_dir)

    if val_data is not None:
        if withTrain:
            args.conf_thresh = 0.01
            args.nms_thresh  = 0.45
        args.scale = False
        
        val_dataset = Custom_yolo_dataset(data = val_data, train=False)
        dataset_size = len(val_dataset)
        num_classes = _num_classes
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
        
        print("\nCalled with args: ")
        print(args)

    else:
        args.scale = True
        args.dataset = "voc07test"
        args.conf_thresh = 0.001
        args.nms_thresh = 0.45
        # args.data_limit = 16
        # print(args)

        # prepare dataset

        if args.dataset == 'voc07trainval':
            args.imdbval_name = 'voc_2007_trainval'

        elif args.dataset == 'voc07test':
            args.imdbval_name = 'voc_2007_test'

        else:
            raise NotImplementedError

        val_imdb = get_imdb(args.imdbval_name)

        val_dataset = RoiDataset(val_imdb, train=False)
        dataset_size = len(val_imdb.image_index)
        num_classes = val_imdb.num_classes
        all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    
    if not args.data_limit==0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))

    args.output_dir = temp_path #check argument parser
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #model = model    ### ??????
    model = Yolov2
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    # model = Yolov2()
    # weight_loader = WeightLoader()
    # weight_loader.load(model, 'yolo-voc.weights')
    # print('loaded')

    model_path = os.path.join(args.output_dir, 'weights.pth')
    torch.save({'model': model.state_dict(),} , model_path)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # print(f'Model loaded from {model_path}')

    if args.use_cuda:
        model.cuda()

    model.eval()
    print("\n Model Loaded ! \n")

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos, paths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Performing validation.", ascii=' ~'):
        # for batch, (im_data, im_infos) in enumerate(val_dataloader):
        # for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                if customData is True:
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh, scale=args.scale)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    if customData is False:
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 5))
                                cls_det[:, :4] = detections[inds, :4]
                                cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                                all_boxes[cls][img_id] = cls_det.cpu().numpy()
                    elif customData is True:
                        _detAllclass = []
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_ = torch.zeros(inds.numel())
                                cls_[:] = cls
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = cls_
                                cls_det[:, 2:6] = detections[inds, :4]
                                cls_det[:, 1] = detections[inds, 4] * detections[inds, 5]
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class)
                        with open(f'{save_dir}/{name}', 'w') as f:
                            f.writelines(_detAllclass)                                    
    if customData and withTrain:
        args.gtFolder = '/home/zafar/old_pc/data_sets/BDD_dataset/Bdd_uncleaned/3class_bdd/val'
        args.detFolder = save_dir
        args.iouThreshold = 0.5
        args.gtFormat = 'xywh'
        args.detFormat = 'xywh'
        args.gtCoordinates = 'rel'
        args.detCoordinates = 'rel'
        args.imgSize = '1280,720'   # for bdd validation data
        args.savePath = './output/plots'
        args.call_with_train = True
        args.showPlot = False
        map, class_metrics = pascalvoc.main(args)
    elif customData and not withTrain:
        map, class_metrics = pascalvoc.main(args)    
    else:
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # map = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
        map = val_imdb.evaluate_detections_with_train(all_boxes, output_dir=args.output_dir)
    return map, class_metrics



if __name__ == '__main__':
    test_for_train()

