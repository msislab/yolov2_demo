import socket
import os
import argparse
# from os import listdir
from re import findall
import time
import shutil
from pseudoLabel_recovery import parse_arguments, recovery

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

def start_server():
    '''
    starts the srever instance at one machine probably D4'''
    # buff_size = 1024

    #initiate connection    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = (socket.gethostname(), 2019)  #change here for sending to another machine in LAN
    s.bind(server_addr)
    s.listen(5)
    return s

def transferFiles(client, fileList, dir_path):
    #Send message to client to notify about sending image
    print("Server sending command: \"Start sending image.\"")
    client.sendall(bytes("Start sending image." ,"utf-8"))

    #wait for reply from client
    print("Server is now waiting for acknowledge from client.")
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')

    #Send message to client to notify about sending image
    Count = len(fileList)
    print("Server sends the number of images to be transfered to client.")
    client.sendall(bytes(str(Count) ,"utf-8"))

    #wait for reply from client
    print("Server is now waiting for acknowledgement from client.")
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge img count.')     

    print("Server will now send the Files.")
    for file in fileList:
        
        print("Server sending command: \"Start sending file name.\"")
        client.sendall(bytes("Start sending file name." ,"utf-8"))
        ack_from_client = wait_for_acknowledge(client,"ACK")
        if ack_from_client != "ACK":
            raise ValueError('Client does not acknowledge command.')
        client.sendall(bytes(file ,"utf-8"))
        
        _file = open(os.path.join(dir_path,file), 'rb')
        b_file = _file.read()
        size = len(b_file)        
        client.sendall(bytes(str(size) ,"utf-8"))
        print(f"\t sending image {file} size of {size}B.")
        
        # print("Server is now waiting for acknowledge from client.")
        ack_from_client = wait_for_acknowledge(client,"ACK")
        if ack_from_client != "ACK":
            raise ValueError('Client does not acknowledge img size.')
        client.sendall(b_file)
        _file.close()
        print(f"{file} is sent!")
        
        # print("Server is now waiting for acknowledge from client.")
        ack_from_client = wait_for_acknowledge(client,"ACK")
        if ack_from_client != "ACK":
            raise ValueError('Client does not acknowledge image transfer completion.')

    print("All files sent.")
    # client.close()

def recieve(client, save_dir):
    ## prepare for recieving High conf pred files
    cmd_from_client = wait_for_acknowledge(client, "Start sending pred files.")
    if cmd_from_client == "Start sending pred files.":
        print("Command \"Start sending txt files.\" Acknowledged...")
        client.sendall(bytes("ACK","utf-8"))
        try:
            print("server is waiting for file number...")
            txtCount_from_client = int(wait_for_acknowledge(client,str(3)))          
        except:
            raise ValueError("Number of images received is buggy.")
    
    if txtCount_from_client > 0:
        print("Number of files to receive: ", txtCount_from_client)
        print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))
        
    print(f"Server is now receiving {txtCount_from_client} files.")
        
    # Receiving files
    recieveFiles(txtCount_from_client, client, save_dir)

def recieveFiles(count, client, save_dir):
    
    # check and make save dir at client side
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    for i in range(count):
    
        print(f"\tReceiving file name")
        cmd_from_client = wait_for_acknowledge(client,"Start sending file name.")
        if cmd_from_client == "Start sending file name.":
            print("Command \"Start sending file name.\" received.")
            print("Sending ACK...")
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
        print(f"\tReceiving text file {index} with name {name}")
        fileSize = int(wait_for_acknowledge(client,str(3)))
        print(f"\tFile size of {fileSize}B received by Server")
        print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))  
        amnt_data = 0
        data = b""
        while amnt_data < fileSize:
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

def adjustDataPath_list(dir_path, fileList):
    for i in range(len(fileList)):
        fileList[i] = os.path.join(dir_path, fileList[i])
    return fileList    

def arguments():
    parser = argparse.ArgumentParser(description='Pseudo-labelling')
    parser.add_argument('--unlabeled_dataPath', type=str,
                        default='/home/zafar/yolov2_demo/data/unlabeled_data',
                        help='Provide the dir path of the unlabeled data')
    parser.add_argument('--GT_dataPathTrain', type=str,
                        default='/home/zafar/yolov2_demo/data/GT_data',
                        help='Provide the dir path of the labeled data')
    parser.add_argument('--GT_dataPathVal', type=str,
                        default='/home/zafar/yolov2_demo/data/val',
                        help='Provide the dir path of the labeled data')
    
    args, unknown = parser.parse_known_args()
    return args

def main():
    Args = arguments()
    # dir_path  = '/home/zafar/yolov2_demo/unlabeld_data'

    fileList_unlabeled = [file for file in os.listdir(Args.unlabeled_dataPath) if findall(r'.jpg',file) != []]  #include all .jpg photos in that directory
    
    img_GTtrain       = [file for file in os.listdir(Args.GT_dataPathTrain) if findall(r'.jpg',file) != []]
    annot_GTtrain     = [file for file in os.listdir(Args.GT_dataPathTrain) if findall(r'.txt',file) != []]
    List_labeledTrain = img_GTtrain + annot_GTtrain

    img_labeled_GTVal   = [file for file in os.listdir(Args.GT_dataPathVal) if findall(r'.jpg',file) != []]
    annot_labeled_GTVal = [file for file in os.listdir(Args.GT_dataPathVal) if findall(r'.txt',file) != []]
    List_labeledVal     = img_labeled_GTVal + annot_labeled_GTVal
    #fileList = ['jihyo.jpg','dami.jpg','uju.jpg']   #images to be sent over to client
    # print(fileList)
    s = start_server()
    # connect with client
    client, address = s.accept()
    print(f"Connection from {address} has been established!")

    client.sendall(bytes("Wait for your task","utf-8"))

    # Transfer unlabeled data to client
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client == "ACK":
        client.sendall(bytes("sending data for pseudo-labelling","utf-8"))

    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client == "ACK":
        transferFiles(client, fileList_unlabeled, Args.unlabeled_dataPath)
    
    # Assign model inference task to client
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')
    else:
        client.sendall(bytes("Perform inference" ,"utf-8"))
        
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')
    
    # get back the model prediction files
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client == "ACK":
        client.sendall(bytes("Send back High and Low conf Preds" ,"utf-8"))
    
    # server now recieves back prediction files from the client
    cmd_from_client = wait_for_acknowledge(client, "sending model predictions to server.")
    if cmd_from_client == "sending model predictions to server.":
        print("Server is ready to recieve prediction files")
        # print("Sending ACK...")
        client.sendall(bytes("ACK","utf-8"))
        time.sleep(0.1)
          
    ##-----------------------------------------------------------------------------------------------
    # Recieving High conf predictions
    cmd_from_client = wait_for_acknowledge(client, "sending High conf preds")
    if cmd_from_client == "sending High conf preds":
        save_dir  = os.path.join(os.getcwd(),'HC')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        client.sendall(bytes("ACK","utf-8"))
    
    recieve(client, save_dir)
    print("All High conf files received...")
    client.sendall(bytes("ACK","utf-8"))
    
    ##--------------------------------------------------------------------------------------------------
    # Recieving Low conf predictions
    cmd_from_client = wait_for_acknowledge(client, "Sending Low conf preds")
    if cmd_from_client == "Sending Low conf preds":
        save_dir  = os.path.join(os.getcwd(),'LC')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        client.sendall(bytes("ACK","utf-8"))
    
    recieve(client, save_dir)

    ##------------------------------------------------------------------------------------
    print("All files received.")
    client.sendall(bytes("ACK","utf-8"))

    cmd_from_client = wait_for_acknowledge(client, "ACK")
    if cmd_from_client=='ACK':
        client.sendall(bytes("Wait for starting self training","utf-8"))

    ##--------------------------------------------------------------------------------------
    
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')                        
    
    client.sendall(bytes("ACK","utf-8"))
    
    ## Perform pseudo-label recovery and prepare self training data for client to start self-tarining
    # print("Closing connection now.")
    # client.close()
    print('Performing pseudo-label recovery...')
    ## prepare for forward and backward tarcking
    dataPath_list = adjustDataPath_list(Args.unlabeled_dataPath, fileList_unlabeled)
    args = parse_arguments()
    args.output_dir     = os.path.join(os.getcwd(),'output/pseudo_labels')
    args.low_conf_path  = os.path.join(os.getcwd(),'LC')
    args.high_conf_path = os.path.join(os.getcwd(),'HC')
    args.data           = dataPath_list

    pseudo_labelPath = recovery(args)

    print('pseudo-label recovery is done...')
    # client.sendall(bytes("ACK","utf-8"))
    
    # sending all self-training data (GT + pseudo-label data) to client for training
    print("Sending self training data to client.")
    time.sleep(0.1)
    client.sendall(bytes("sending self training data.","utf-8"))
    ack_from_client = wait_for_acknowledge(client,"ACK")
    ##sending GT data for training
    if ack_from_client=="ACK":
        client.sendall(bytes("sending GT data.","utf-8"))
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client=="ACK":    
        transferFiles(client, List_labeledTrain, Args.GT_dataPathTrain)
    
    ## sending pseudo-label data for training
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client=="ACK":
        client.sendall(bytes("sending Pseudo-label data","utf-8"))
    ack_from_client = wait_for_acknowledge(client,"ACK")    
    if ack_from_client=="ACK":
        imgList_pseudolabeled   = [file for file in os.listdir(pseudo_labelPath) if findall(r'.jpg',file) != []]
        annotList_pseudolabeled = [file for file in os.listdir(pseudo_labelPath) if findall(r'.txt',file) != []]
        fileList_pseudolabeled  = imgList_pseudolabeled + annotList_pseudolabeled
        transferFiles(client, fileList_pseudolabeled, pseudo_labelPath)
    
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')
    
    if ack_from_client=="ACK":
        client.sendall(bytes("sending validation data.","utf-8"))
    ack_from_client = wait_for_acknowledge(client,"ACK")    
    if ack_from_client=="ACK":
        transferFiles(client, List_labeledVal, Args.GT_dataPathVal)
    
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')    
    
    client.sendall(bytes("Train the model on sent data.","utf-8"))

    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')
    
    ack_from_client = wait_for_acknowledge(client,"ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge command.')
    print()
    
if __name__ == '__main__':
    main()