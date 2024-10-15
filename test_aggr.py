# Importing Libraries
import os
import torch
import numpy as np 
import copy
import argparse
from collections import OrderedDict

# Importing Functions
from yolov2_pytorchClient.yolov2_tiny_2 import Yolov2
from yolov2_pytorchClient.Test_with_train import test_for_train
from yolov2_pytorchClient.util.data_util import check_dataset


def main(Args):

    # YAML file
    c0_output_dir = "C1_Output"
    c1_output_dir = "C2_Output"
    c2_output_dir = "C3_Output"
    c3_output_dir = "C4_Output"
    _output_dir   = "Aggr_Output"
    
    #Checking YAML File
    data_dict = check_dataset(Args.data)
    _, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
    Args.val_dir = val_dir
    names = data_dict['names']

    # Weights Path
    client_0_path = "/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth"
    client_1_path = "/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth"
    client_2_path = "/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth"
    client_3_path = "/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth"
    # client_3_path = "/data/DOCKER_DIRS/ali5021/Yolov5_DeepSORT_PseudoLabels/yolov2_pytorch/data/basline_itr/yolov2_bl_48_35_map.pth"

    # Pre-trained Weights
    # PT_weights    = "/data/DOCKER_DIRS/ali5021/Yolov5_DeepSORT_PseudoLabels/yolov2_pytorch/data/basline_itr/yolov2_bl_48_35_map.pth"
    # PT_weights    = "/home/gpuadmin/ZPD/Yolov5_DeepSORT_PseudoLabels/FED_DIF_DATA_E1/yolov2_client1_R3_best_map.pth"
    # PT_weights    = "/home/gpuadmin/ZPD/Yolov5_DeepSORT_PseudoLabels/FED_DIF_DATA_E1/yolov2_client0_R1_best_map.pth"

    #  Intializing Model   
    client_0_model  =  Yolov2(classes=names)
    client_1_model  =  Yolov2(classes=names)
    client_2_model  =  Yolov2(classes=names)
    client_3_model  =  Yolov2(classes=names)
    aggr_model      =  Yolov2(classes=names)


    # Loading Weights
    pre_trained_checkpoint_c0 = torch.load(client_0_path, map_location = 'cpu')
    client_0_model.load_state_dict(pre_trained_checkpoint_c0['model'])

    pre_trained_checkpoint_c1 = torch.load(client_1_path, map_location = 'cpu')
    client_1_model.load_state_dict(pre_trained_checkpoint_c1['model'])

    pre_trained_checkpoint_c2 = torch.load(client_2_path, map_location = 'cpu')
    client_2_model.load_state_dict(pre_trained_checkpoint_c2['model'])

    pre_trained_checkpoint_c3 = torch.load(client_3_path, map_location = 'cpu')
    client_3_model.load_state_dict(pre_trained_checkpoint_c3['model'])

    # aggr_model.load_state_dict(pre_trained_checkpoint_c3['model'])

    # pre_trained_aggr = torch.load(PT_weights, map_location = 'cpu')
    # aggr_model.load_state_dict(pre_trained_aggr['model'])

    mAP, _ = test_for_train(f'{Args.output_dir}/client', client_3_model.to('cpu'),
                            Args, val_path,
                            names, True,
                            device=Args.device)
    
    print('Client mAP: ', mAP)


    mAP, _ = test_for_train(f'{Args.output_dir}/Aggr', aggr_model.to('cpu'),
                            Args, val_path,
                            names, True,
                            device=Args.device)
    
    print('Aggr mAP: ', mAP)

    print("\nValidating All the Weights before Aggregation:")
    models=[]
    
    print("\nValidating Client_0 before Aggregation:")    
    # _mAP, _ = test_for_train(temp_path=c0_output_dir, model=client_0_model, args=Args, val_data=val_path, classes=names, afterTrain=True, device=Args.device)
    models.append(client_0_model)
    
    
    print("\nValidating Client_1 before Aggregation:")    
    # _mAP, _ = test_for_train(temp_path=c1_output_dir, model=client_1_model, args=Args, val_data=val_path, classes=names, afterTrain=True, device=Args.device)
    models.append(client_1_model)
    
    print("\nValidating Client_2 before Aggregation:")    
    # _mAP, _ = test_for_train(temp_path=c2_output_dir, model=client_2_model, args=Args, val_data=val_path, classes=names, afterTrain=True, device=Args.device)
    models.append(client_2_model)
    
    print("\nValidating Client_3 before Aggregation:")    
    # _mAP, _ = test_for_train(temp_path=c3_output_dir, model=client_3_model, args=Args, val_data=val_path, classes=names, afterTrain=True, device=Args.device)
    models.append(client_3_model)
# Aggregate weights and update (to self.shared_memory['aggr_weights'])
    _aggrModel = copy.deepcopy(aggr_model)
    # for name, param in _aggrModel.named_parameters():
    newAggr = []
    for i, (name, param) in enumerate(_aggrModel.state_dict().items()):
        print(f"Aggregating {name}")
        # agg_state = aggr_model.state_dict()
        data= []
        for j,_m in enumerate(models):
            _m_state = list(_m.state_dict().items())
            _, _param = _m_state[i]
            data.append(_param)
        meanData = np.mean(np.array(data), axis=0)
        if isinstance(meanData, float):
            newAggr.append((name, torch.tensor(meanData)))
        else:
            newAggr.append((name, torch.from_numpy(meanData)))    
    newAggr = OrderedDict(newAggr)    
        

            
    # aggState = newAggr.state_dict()
    aggr_model.load_state_dict(newAggr)

    aggr_model.eval()
    
    out_dir = f'{Args.output_dir}/server'
    print(f'Validating Aggregated server weights...')
    Aggr_mAP, _ = test_for_train(out_dir, aggr_model.to('cpu'),
                                Args, val_path,
                                names, True,
                                device=Args.device)
    print()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--batch_size', dest='batch_size',
                        default=32, type=int)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        default='voc0712trainval', type=str)
    parser.add_argument('--data', type=str, dest='data',
                        default="/home/zafar/yolov2_demo/data.yaml", help='Give the path of custom data .yaml file' )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='/home/zafar/yolov2_demo/yolov2_pytorchClient/Aggr_Output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=False, type=bool)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--weights', default='', dest='weights',
                        help='provide the path of weight file (.pth) if resume')
    parser.add_argument('--device', type= int,
                        default=1, dest='device',
                        help='Choose a gpu device 0, 1, 2 etc.')
    parser.add_argument('--savePath', default='results',
                        help='')
    parser.add_argument('--imgSize', default='1280,720',
                        help='image size w,h') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.dataset = 'custom'
    main(args)