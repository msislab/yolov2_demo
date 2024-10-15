import os
import argparse
import yaml
import shutil
import re
from yolov2_pytorch.demo import demo
from yolov2_pytorch.demo import parse_args as _parse_args
from pseudoLabel_recovery import recovery
from yolov2_pytorchClient import train_yolov2_tiny
import test_client

def arguments():
    parser = argparse.ArgumentParser(description='Pseudo-labelling')
    parser.add_argument('--save_dir', type=str,
                        default=f'{os.getcwd()}',
                        help='Provide the dir path of the unlabeled data')
    parser.add_argument('--model_path', type=str,
                        default='/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth',
                        help='Provide the dir path of the labeled data')
    parser.add_argument('--dataPath', type=str,
                        default='/home/zafar/yolov2_demo/data',
                        help='path to data parent dir')
    parser.add_argument('--labelData', type=str,
                        default='/home/zafar/yolov2_demo/label.txt',
                        help='path to label.txt')
    parser.add_argument('--valData', type=str,
                        default='/home/zafar/yolov2_demo/val.txt',
                        help='path to val.txt')
    parser.add_argument('--val_dir', type=str,
                        default='/home/zafar/yolov2_demo/data/val',
                        help='path to val data dir')
    parser.add_argument('--unlabelData', type=str,
                        default='/home/zafar/yolov2_demo/data/unlabeled_data',
                        help='path to unlabel data dir')
    
    args, unknown = parser.parse_known_args()
    return args

def makeDataFile(dir, save=False):
    
    lines = []
    for file in os.listdir(dir):
        if re.findall(r'.jpg',file) != []:
            if not save:
                path = os.path.join(dir, file)
                lines.append(path)
            else:
                path = os.path.join(dir, file) + '\n'
                lines.append(path)    
    if save:
        if os.path.isfile(os.path.join(os.getcwd(),'unlabeled_data.txt')):
            os.remove(os.path.join(os.getcwd(),'unlabeled_data.txt'))
            with open(os.path.join(os.getcwd(),'unlabeled_data.txt'), 'w') as f:
                f.writelines(lines)
        else:
            with open(os.path.join(os.getcwd(),'unlabeled_data.txt'), 'w') as f:
                f.writelines(lines)
        return os.path.join(os.getcwd(),'unlabeled_data.txt')
    else:
        return lines

def makeYAML(args, nc, names):
    
    # make yaml file for ith client
    data = {}
    data['path' ]    =  args.dataPath
    data['train']    =  args.trainPath
    data['val'  ]    =  args.valPath
    data['test'  ]   =  args.valPath
    data['val_dir']  =  args.valDir
    data['nc'   ]    =  nc
    data['names']    =  names
    with open(f'{os.getcwd()}/data.yaml', 'w+') as file:
        yaml.dump(data,file)
    return f'{os.getcwd()}/data.yaml'

def main():

    # setting main arguments

    _client = test_client.start_client()
    checkpoint, trainArgs = test_client.main(_client)

    # Trainingdata_path = makeYAML(Args.save_dir, trainArgs['nc'], trainArgs['classes'])
    Args = arguments()
    Args.save_dir = os.path.join(Args.save_dir,'pseudo_data')
    if os.path.exists(Args.save_dir):
        shutil.rmtree(Args.save_dir, ignore_errors=True)
        os.makedirs(Args.save_dir)
    else:
        os.makedirs(Args.save_dir)

    args = _parse_args()
        
        # setup necessary args   
    args.data       = makeDataFile(Args.unlabelData, save=True)
    # args.model_name = Args.model_path
    
    # low conf pseudo-label generation
    low_conf_dir     = os.path.join(os.getcwd(),'output/pseudo_labels/low_conf')  
    args.pseudo_type = 'low_conf'
    args.output_dir  = low_conf_dir
    args.classes     = ['Vehicle']
    args.weights     = checkpoint
    demo(args)
    
    # high conf pseudo-label generation
    high_conf_dir = os.path.join(os.getcwd(),'output/pseudo_labels/high_conf')
    args.pseudo_type = 'high_conf'
    args.output_dir  = high_conf_dir
    args.classes     = ['Vehicle']
    # args.weights     = checkpoint
    demo(args)

    # Perform pseudolabel recovery
    # set necessary args for recovery
    args.output_dir     = 'output/pseudo_labels'
    args.low_conf_path  = low_conf_dir
    args.high_conf_path = high_conf_dir
    args.use_cuda       = True
    args.device         = '0'
    args.data           = makeDataFile(Args.unlabelData)
    # print(args)
    pseudoLabel_path = recovery(args=args, model=args.model_name)
    with open(Args.labelData, 'r') as f:
        GTdata = f.readlines()
    pseudoData = [file for file in os.listdir(pseudoLabel_path) if re.findall(r'.jpg',file) != []]
    train_data = GTdata + pseudoData
    path       = f'{os.getcwd()}/train.txt'
    with open(f'{path}', 'w') as f:
        f.writelines(train_data)        


    args.dataPath     = Args.dataPath
    args.trainPath    = path
    args.valPath      = Args.valData
    args.valDir       = Args.val_dir
    Trainingdata_path = makeYAML(args, trainArgs['nc'], trainArgs['classes'])

    _args = train_yolov2_tiny.parse_args()
    _args.dataset    = 'custom'
    _args.data       = Trainingdata_path
    _args.resume     = True
    _args.weights    = checkpoint
    _args.cleaning   = True
    _args.output_dir = os.path.join(os.getcwd(), 'Training_output')
    _args.max_epochs = trainArgs['rounds'] * trainArgs['epsPerR']
    _args.device     = '0'

    print('Starting training...')
    _client.sendall(bytes("Self-Training started.","utf-8"))
    train_yolov2_tiny.train(_args, _client, True, trainArgs['epsPerR'])
    print('Training is finished')
    # client.sendall(bytes("ACK", "utf-8"))
    print()
    # print("Closing connection.")
    _client.close()
    
if __name__ == '__main__':
    # model_path = '/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2-tiny-voc.pth'
    # save_dir = os.path.join(os.getcwd(),'server_data')
    main() 