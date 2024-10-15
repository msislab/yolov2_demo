# Importing Libraries
import os
import torch
import numpy as np 
import copy
import argparse

# Importing Functions
from yolov2_tiny_2 import Yolov2
from Test_with_train import test_for_train
from util.data_util import check_dataset


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

    # pre_trained_aggr = torch.load(PT_weights, map_location = 'cpu')
    # aggr_model.load_state_dict(pre_trained_aggr['model'])

    mAP, _ = test_for_train(f'{Args.output_dir}/client', client_3_model.to('cpu'),
                            Args, val_path,
                            names, True,
                            device=Args.device)
    
    print('Client mAP: ', mAP)

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
    for name, param in aggr_model.named_parameters():
        print(f"Aggregating {name}")
        # agg_state = aggr_model.state_dict()
        data= []
        for i,_m in enumerate(models):
            _m_state = _m.state_dict()
            data.append(_m_state[name].detach().cpu().numpy())
        aggr_data = np.mean(np.array(data), axis=0)
        param.data = torch.from_numpy(aggr_data)
        if (param.data).detach().cpu().numpy() == aggr_data:print("Equal")
        else:print("False")
        
        # for i in range()
        # param = torch.from_numpy(aggr_data)
        # agg_state[name] = torch.from_numpy(aggr_data)
        
    
    # aggr_model.load_state_dict(agg_state['model'])
    
    
    # pretrained_checkpoint = torch.load(_path,map_location='cpu')
    # loaded_model = torch.load(_path,map_location='cpu')['model']
    
    # custom_model.load_state_dict(loaded_model)
    # temp_aggr_model = custom_model.state_dict()   
    
    # for name in temp_aggr_model:

    #     if name =='conv1.weight':
    #         temp_aggr_model[name] = agg_state[name]
        

            
    
    
    out_dir = f'{Args.output_dir}/server'
    print(f'Validating Aggregated server weights...')
    Aggr_mAP, _ = test_for_train(out_dir, aggr_model.to('cpu'),
                                Args, val_path,
                                names, True,
                                device=Args.device)


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
                        default='Aggr_Output', type=str)
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