import socket
import os
import re
from yolov2_pytorch.demo import demo
from yolov2_pytorch.demo import parse_args as _parse_args
import test_client
import shutil
# from pseudoLabel_recovery import recovery
import time
import argparse
import yaml
# from utility import wait_for_acknowledge

def wait_for_acknowledge(client,response):
    """
    Waiting for this response to be sent from the other party
    """
    amount_received = 0
    amount_expected = len(response)
    
    msg = str()
    while amount_received < amount_expected:
        data = client.recv(128)
        amount_received += len(data)
        msg += data.decode("utf-8")
        #print(msg)
    return msg

def start_client():
    #initiate connection
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_addr = (socket.gethostname(),2019)  #change here for sending to another machine in LAN
    # server_addr = ('192.168.1.2',5050)
    client.connect(server_addr)
    print(f"Connected to server!")
    return client

def recieveFiles(Count_from_server, client, save_dir):
    
    # check and make save dir at client side
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)
    
    print(f"\tReceiving files")
    for i in range(Count_from_server):
        
        cmd_from_server = wait_for_acknowledge(client,"Start sending file name.")
        if cmd_from_server == "Start sending file name.":
            # print("Command \"Start sending image name.\" received.")
            # print("Sending ACK...")
            client.sendall(bytes("ACK","utf-8"))    
        name = wait_for_acknowledge(client,str(3))
        
        index = i+1
        # file = f"./imgfromserver{index}.jpg"   
        file = os.path.join(save_dir,name)
        try:                                            #check for existing file, will overwrite
            f = open(file, "x")           
            f.close()
        except:
            pass
        finally:
            f = open(file, "wb")
        print(f"\tReceiving file {index} with name {name}")
        size = int(wait_for_acknowledge(client,str(3)))
        # print(f"\tImage size of {imgsize}B received by Client")
        # print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))  
        amnt_data = 0
        data = b""
        while amnt_data < size:
            buff = client.recv(1024)
            amnt_data += len(buff)
            data += buff
        # print(len(data))
        f.write(data)
        f.close()
        print(f"File {file} received!")
        # print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))
        #a = wait_for_acknowledge(client,"This is done.")

def recieve(client, save_dir):

    cmd_from_server = wait_for_acknowledge(client,"Start sending image.")

    #send an ACK 
    Count_from_server = 0
    if cmd_from_server == "Start sending image.":
        print("Command \"Start sending image.\" Acknowledged...")
        # print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))
        try:
            print("Waiting for number count of files to be sent by server")
            Count_from_server = int(wait_for_acknowledge(client,str(3)))          
        except:
            raise ValueError("Number of images received is buggy.")

    if Count_from_server > 0:
        print("Number of images to receive: ",Count_from_server)
        # print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))

    ## recieve data from server
    print(f"Client is now receiving {Count_from_server} images.")
    recieveFiles(Count_from_server, client, save_dir)

def makeDataFile(dir, save=False):
    
    lines = []
    for file in os.listdir(dir):
        if re.findall(r'.jpg',file) != []:
            path = os.path.join(dir, file) + '\n'
            lines.append(path)
    if save:
        if os.path.isfile(os.path.join(os.getcwd(),'server_data.txt')):
            os.remove(os.path.join(os.getcwd(),'server_data.txt'))
            with open(os.path.join(os.getcwd(),'server_data.txt'), 'w') as f:
                f.writelines(lines)
        else:
            with open(os.path.join(os.getcwd(),'server_data.txt'), 'w') as f:
                f.writelines(lines)
        return os.path.join(os.getcwd(),'server_data.txt')
    else:
        return lines

def transferFiles(client, fileList, dir_path):
    #Send message to server to notify about sending files
    print("Client sending command: \"Start sending text files.\"")
    client.sendall(bytes("Start sending pred files." ,"utf-8"))

    #wait for reply from server
    # print("Client is now waiting for server acknowledgment.")
    ack_from_server = wait_for_acknowledge(client,"ACK")
    if ack_from_server != "ACK":
        raise ValueError('Server does not acknowledge command.')

    #Send message to server to notify about sending the number of files
    Count = len(fileList)
    print("Sending file number to be sent...")
    client.sendall(bytes(str(Count) ,"utf-8"))

    #wait for reply from server
    # print("Client is now waiting for server acknowledgment.")
    ack_from_server = wait_for_acknowledge(client,"ACK")
    if ack_from_server != "ACK":
        raise ValueError('Server does not acknowledge command.')  

    print("Client will now send the text files.")
    for file in fileList:
        
        # print("Client sending command: \"Start sending pseudo-labelels.\"")
        client.sendall(bytes("Start sending file name." ,"utf-8"))
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server != "ACK":
            raise ValueError('Server does not acknowledge command.')
        client.sendall(bytes(file ,"utf-8"))
        
        _file = open(os.path.join(dir_path,file), 'rb')
        b_file = _file.read()
        Size = len(b_file)        
        client.sendall(bytes(str(Size) ,"utf-8"))
        print(f"\t sending file {file} size of {Size}B.")
        
        # print("Client is now waiting for server acknowledgment.")
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server != "ACK":
            raise ValueError('Client does not acknowledge img size.')
        client.sendall(b_file)
        _file.close()
        print(f"{file} is sent!")
        
        # print("Client is now waiting for server acknowledgment.")
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server != "ACK":
            raise ValueError('Client does not acknowledge image transfer completion.')

    print("All files sent...")

def makeYAML(data_path, nc, names):
    # make yaml file for ith client
    data = {}
    data['path' ]    =  data_path
    data['train']    =  os.path.join(data_path, 'train.txt')
    data['val'  ]    =  os.path.join(data_path, 'val.txt')
    data['test'  ]    =  os.path.join(data_path, 'val.txt')
    data['val_dir']  =  os.path.join(data_path, 'val_data')
    data['nc'   ]    =  nc
    data['names']    =  names
    with open(f'{data_path}/data.yaml', 'w+') as file:
        yaml.dump(data,file)
    return f'{data_path}/data.yaml'   

def arguments():
    parser = argparse.ArgumentParser(description='Pseudo-labelling')
    parser.add_argument('--save_dir', type=str,
                        default=f'{os.getcwd()}',
                        help='Provide the dir path of the unlabeled data')
    parser.add_argument('--model_path', type=str,
                        default='/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2_best_map.pth',
                        help='Provide the dir path of the labeled data')
    
    args, unknown = parser.parse_known_args()
    return args

def main():

    # setting main arguments
    Args = arguments()
    Args.save_dir = os.path.join(Args.save_dir,'server_data')

    if os.path.exists(Args.save_dir):
        shutil.rmtree(Args.save_dir, ignore_errors=True)
        os.makedirs(Args.save_dir)
    else:
        os.makedirs(Args.save_dir)    

    # start client
    client = start_client()              # D4 as server for pseudo-labeling
    _client = test_client.start_client() # aggregation server on 41
    checkpoint, trainArgs = test_client.main(_client)
    # client.settimeout(5) #limit each communication time to 5s
    #listening to server command
    print("Client is now waiting for server's command.")

    cmd_from_server = wait_for_acknowledge(client,"Wait for your task")
    if cmd_from_server == "Wait for your task":
        print('Waiting for assignment from server...')
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.5)

    ## Prepare for recieving data from server
    cmd_from_server = wait_for_acknowledge(client,"sending data for pseudo-labelling")
    if cmd_from_server == "sending data for pseudo-labelling":
        save_dir = os.path.join(Args.save_dir,'unlabeled_data')
        print('Ready to recieve data...')
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.2)
    
    # Recieve the unlabeled data from server
    recieve(client, save_dir)

    print("All images received...")
    client.sendall(bytes("ACK","utf-8"))

    ## prepare for new task that is model inference for pseudo-labelling
    print('Waiting for assignment from server...')
    cmd_from_server = wait_for_acknowledge(client,"Perform inference")
    if cmd_from_server == 'Perform inference':
        print("Command \"Perform inference\" Acknowledged...")
        client.sendall(bytes("ACK","utf-8"))
        args = _parse_args()
        
        # setup necessary args   
        args.data       = makeDataFile(save_dir, save=True)
        args.model_name = Args.model_path
        
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
        demo(args)
        
        # # Perform pseudolabel recovery
        # # set necessary args for recovery
        # args.output_dir     = 'output/pseudo_labels'
        # args.low_conf_path  = low_conf_dir
        # args.high_conf_path = high_conf_dir
        # args.use_cuda       = True
        # args.device         = '0'
        # # print(args)
        # pseudoLabel_path = recovery(args=args, model=args.model_name)
        
    ## prepare for new job
    print('Model inference done...')
    client.sendall(bytes("ACK","utf-8"))
    print('Waiting for assignment from server...')

    # sending back model predictions
    cmd_from_server = wait_for_acknowledge(client,"Send back High and Low conf Preds")
    if cmd_from_server=='Send back High and Low conf Preds':
        client.sendall(bytes("sending model predictions to server." ,"utf-8"))

    # high conf pred transfer
    cmd_from_server = wait_for_acknowledge(client,"ACK")
    if cmd_from_server=='ACK':
        fileList_HC = [file for file in os.listdir(high_conf_dir) if re.findall(r'.txt',file) != []]
        fileList_LC = [file for file in os.listdir(low_conf_dir) if re.findall(r'.txt',file) != []]
        
        client.sendall(bytes("sending High conf preds" ,"utf-8"))
        
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server == "ACK":
            time.sleep(0.5)
            transferFiles(client, fileList_HC, high_conf_dir)
        
        # low conf pred transfer
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server == "ACK":
            client.sendall(bytes("Sending Low conf preds","utf-8"))
        ack_from_server = wait_for_acknowledge(client,"ACK")
        if ack_from_server == "ACK":
            time.sleep(0.5)
            transferFiles(client, fileList_LC, low_conf_dir)
        ack_from_server = wait_for_acknowledge(client,"ACK")
        
        if ack_from_server != "ACK":
            raise ValueError('Client does not acknowledge command.')
    
    
    client.sendall(bytes("ACK","utf-8"))
    cmd_from_server = wait_for_acknowledge(client, "Wait for starting self training")
    if cmd_from_server=='Wait for starting self training':
        client.sendall(bytes("ACK","utf-8"))
        print('Wait for starting self training...')
        # time.sleep()
    
    cmd_from_server = wait_for_acknowledge(client, "ACK")
    if cmd_from_server != "ACK":
        raise ValueError('Client does not acknowledge command.')
    
    # preparing to recieve self-training data for training (GT_labeled data) 
    # cmd_from_server = wait_for_acknowledge(client, "ACK")
    # if cmd_from_server == "ACK":
    #     time.sleep(5)
    # print('recieving ST data')
    time.sleep(1)
    cmd_from_server = wait_for_acknowledge(client, "sending self training data.")
    if cmd_from_server=="sending self training data.":
        print('Ready to recieve data')
        client.sendall(bytes("ACK","utf-8"))

    # recieving GT data
    cmd_from_server = wait_for_acknowledge(client, "sending GT data.")
    if cmd_from_server=="sending GT data.":
        save_dir = os.path.join(Args.save_dir,'GT_data')
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.1)
    # recieve GT data from server    
    recieve(client, save_dir)
    
    client.sendall(bytes("ACK","utf-8"))
    # recieving pseudolabel data
    cmd_from_server = wait_for_acknowledge(client, "sending Pseudo-label data")
    if cmd_from_server=="sending Pseudo-label data":
        save_dir = os.path.join(Args.save_dir,'pseudoLabel_data')
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.1)
    recieve(client, save_dir)

    client.sendall(bytes("ACK","utf-8"))
    # recieving validation data
    cmd_from_server = wait_for_acknowledge(client, "sending validation data.")
    if cmd_from_server=="sending validation data.":
        save_dir = os.path.join(Args.save_dir,'val_data')
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.1)
    recieve(client, save_dir)

    client.sendall(bytes("ACK","utf-8"))

    cmd_from_server = wait_for_acknowledge(client, "Train the model on sent data.")
    if cmd_from_server == "Train the model on sent data.":
        client.sendall(bytes("ACK", "utf-8"))
        print("Client is preparing for training on the data sent by Server..")
        GT_data = makeDataFile(os.path.join(Args.save_dir,'GT_data'))
        pseudoLabel_data = makeDataFile(os.path.join(Args.save_dir,'pseudoLabel_data'))
        data = GT_data + pseudoLabel_data
        with open(os.path.join(Args.save_dir,'train.txt'), 'w') as ff:
            ff.writelines(data)
        Val_data = makeDataFile(os.path.join(Args.save_dir,'val_data'))
        with open(os.path.join(Args.save_dir,'val.txt'), 'w') as ff:
            ff.writelines(Val_data)
        
        # Distributed Self-Training
        from yolov2_pytorchClient import train_yolov2_tiny

        Trainingdata_path = makeYAML(Args.save_dir, trainArgs['nc'], trainArgs['classes'])       
        
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
    client.sendall(bytes("ACK", "utf-8"))
    print()
    # print("Closing connection.")
    # client.close()
    
if __name__ == '__main__':
    # model_path = '/home/zafar/yolov2_demo/yolov2_pytorch/data/pretrained/yolov2-tiny-voc.pth'
    # save_dir = os.path.join(os.getcwd(),'server_data')
    main()    