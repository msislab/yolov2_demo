import socket
import os
import argparse
# from os import listdir
from re import findall
import time
import shutil
import multiprocessing
from multiprocessing import Process, Manager, freeze_support
import torch
from yolov2_pytorchClient.yolov2_tiny_2 import Yolov2
from yolov2_pytorchClient.util.data_util import check_dataset
from yolov2_pytorchClient.Test_with_train import test_for_train, parse_args
import pickle
import struct
# import zlib
import copy
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
import numpy as np
import contextlib

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

def weightAgg(sharedMem, args, Lock):
    # time.sleep(5)
    # setup necessary validation arguments
    data_dict = check_dataset(args.data)
    _, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
    args.val_dir = val_dir
    names   = data_dict['names']
    status  = True
    out_dir = os.path.join(os.getcwd(),copy.deepcopy(args.output_dir))
    rounds  = 0
    _bestmAP    = 0
    
    while status:
        time.sleep(5)
        ids, addrs = [], []
        clientDict = sharedMem['clientIDs']
        if clientDict:
            for id, adr in clientDict.items():
                ids.append(id)
                addrs.append(adr)
        # get client ids
        # print(f'waiting for all clients to finish round {rounds}...')
        while len(ids)==sharedMem['numClients']:
            # clientIds = sharedMem['clientIDs']
            check = [False for x in ids]      # variable which will tell when to perform aggregation
            for i, clientId in enumerate(ids):
                time.sleep(1)
                if sharedMem[f'{clientId}_weights']!=0:
                    check[i] = True
            if all(check):
                print(f'{Style.BRIGHT}All clients have sent checkpoints after round {rounds}...')
                print(f'{Style.BRIGHT}{Fore.YELLOW}aggregating client weights...')
                rounds += 1
                # validate all client models before aggregation
                models = []
                mAPs   = []
                for i, clientId in enumerate(ids):
                    print(f'{Style.BRIGHT} Validating client:{clientId}; address:{addrs[i]}...')
                    _out_dir = f'{out_dir}/{clientId}'
                    _model = copy.deepcopy(sharedMem[f'{clientId}_weights'])
                    with Lock:
                        sharedMem[f'{clientId}_weights'] = 0
                    models.append(_model)
                    # mAPs.append(self.shared_memory[f'weightsClient_{i}'][1])
                    with num_torch_thread(1):
                        _mAP, _ = test_for_train(_out_dir, _model.to('cpu'),
                                                args, val_path,
                                                names, True,
                                                device=0)
                    mAPs.append(_mAP)
                    
                # initiate the aggr_model
                if sharedMem['AggModel'] is None:
                    aggr_model = copy.deepcopy(models[0])
                else:
                    aggr_model = copy.deepcopy(sharedMem['AggModel']).to('cpu')                    
                
                # Aggregate weights and update
                for name, param in aggr_model.named_parameters():
                    print(f"Aggregating {name}")
                    data= []
                    for i,_m in enumerate(models):
                        # mAP = (mAPs[i])*100
                        # w   = mAP/mAP_sum
                        _m_state = _m.state_dict()
                        # _m_state = w * (_m_state[name].detach().cpu().numpy())
                        data.append(_m_state[name].detach().cpu().numpy())
                    aggr_data = np.mean(np.array(data), axis=0)
                    param.data = torch.from_numpy(aggr_data)
                    
                # validate aggregated model
                
                _out_dir = f'{out_dir}/server'
                print(f'{Style.BRIGHT} Validating Aggregated weights...')
                with num_torch_thread(1):
                    Aggr_mAP, _ = test_for_train(_out_dir, aggr_model.to('cpu'),
                                                args, val_path,
                                                names, True,
                                                device=0)
                
                mAPs.append(Aggr_mAP)
                with Lock:
                    sharedMem['AggModel'] = copy.deepcopy(aggr_model).to('cpu')
                
                # log all results here
                
                # find best model and update to sharedMem
                best_map = max(mAPs)
                _index   = mAPs.index(best_map)
                
                if _index==len(ids):
                    best      = aggr_model.to('cpu')
                    save_name = os.path.join(_out_dir, f'yolov2_server_best_aggr_{rounds}.pth')
                    with Lock:
                        sharedMem['bestUpdate'] = copy.deepcopy(best)
                        sharedMem['serverUpdate'] = True
                    print('aggregation server update to shared memory is done...')
                else: 
                    best = models[_index].to('cpu')
                    save_name = os.path.join(_out_dir, f'yolov2_best_client{ids[_index]}.pth')
                    with Lock:
                        sharedMem['bestUpdate'] = copy.deepcopy(best)
                        print('aggregation server update to shared memory is done...')
                        sharedMem['serverUpdate'] = True
                # save best server model
                if best_map > _bestmAP:
                    _bestmAP = best_map
                    with num_torch_thread(1):
                        torch.save({
                            'model': best.state_dict(),
                            }, save_name)
            #update status
            if rounds == sharedMem['totalRounds']:
                status =False
                sharedMem['finish'] = True
                break
    print(f'{Style.BRIGHT}Distributed training ended stopping aggregation process...')                                                          

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

def start_server(numClients):
    '''
    starts the srever instance at one machine probably D4'''
    # buff_size = 1024

    #initiate connection    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = ('', 58316)
    # server_addr = (socket.gethostname(), 2019)  #change here for sending to another machine in LAN
    s.bind(server_addr)
    s.listen(numClients)
    return s

def set4Train(model_path=None):
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = None    
    ars = {"rounds" : 5 ,
           "epsPerR": 5 ,
           "classes": ["Vehicle"],
           "nc": 1}
    return ars, checkpoint

def sendData(client,data):
    serialized = pickle.dumps(data)
    # comp_serialized = zlib.compress(serialized)
    # size = len(serialized)
    client.sendall(struct.pack('>I', len(serialized)))
    cmd_from_client = wait_for_acknowledge(client, "ACK")
    # if len(comp_serialized) > 4096:
    #     chunk_size = 4096
    #     for i in range(0,len(comp_serialized),chunk_size):
    #         client.sendall(data[i:i+chunk_size])
    # # client.sendall(bytes(str(size), "utf-8"))
    # else:
    if cmd_from_client=="ACK":
        print('sending data...')
        client.sendall(serialized)

def recieveData(client):
    # size = int(wait_for_acknowledge(client,str(3)))
    # client.settimeout(10)
    size = struct.unpack('>I', client.recv(4))[0]
    client.sendall(bytes("ACK", "utf-8"))
    print('Recieving weights...')
    amnt_data = 0
    data = b""
    if size > 2000000:
        bfSize = 64000
    else:
        bfSize = 4096
    pbar = tqdm(total=size)    
    while amnt_data < size:
        buff = client.recv(bfSize)
        amnt_data += len(buff)
        data += buff
        pbar.update(len(buff))
    # data = ast.literal_eval(data.decode("utf-8"))
    # data = zlib.decompress(data)    
    data = pickle.loads(data)
    return data

def handle(cli, addr, sharedMem, arg, clientID, Lock):
    # print(f"{Style.BRIGHT}Connection from {Fore.YELLOW}{addr} has been established!")
    cli.sendall(bytes("success...","utf-8"))
    
    # Set up and send distributed training utilities to client
    ack_from_client = wait_for_acknowledge(cli,"ACK")
    if ack_from_client == "ACK":
        print(f'successfuly communicated with {Style.BRIGHT}{addr}')
        print('Sending Distributed self-training setup parameters...')
        cli.sendall(bytes("Sending distributed training args.","utf-8"))
    ack_from_client = wait_for_acknowledge(cli, "ACK")
    if ack_from_client=="ACK":
        trainArgs, checkpoint = set4Train(arg.model_name)
        cli.sendall(bytes("sending args dict.", "utf-8"))
    ack_from_client = wait_for_acknowledge(cli,"ACK")
    if ack_from_client=="ACK":    
        sendData(cli,trainArgs)
        # cli.sendall(bytes(str(trainArgs), "utf-8"))
    ack_from_client = wait_for_acknowledge(cli, "ACK")
    if ack_from_client=="ACK":
        print(f'{Style.BRIGHT}Sending starting checkpoint to {Fore.YELLOW}{addr}...')
        cli.sendall(bytes("sending starting checkpoint.", "utf-8"))
    ack_from_client = wait_for_acknowledge(cli,"ACK")
    if ack_from_client=="ACK":
        sendData(cli,checkpoint["model"])    
    ack_from_client = wait_for_acknowledge(cli,"ACK")
    if ack_from_client!="ACK":
        raise ValueError('Client does not acknowledge command.')
    cmd_from_client = wait_for_acknowledge(cli, "Self-Training started.")
    # client is now training setup server for weight aggregation
    if cmd_from_client=="Self-Training started.":
        print(f'{Style.BRIGHT}{addr} have started training...')
        with Lock:
            sharedMem[f'{clientID}_weights'] = 0      # weights initialization for current client in the memory
        clientDict = sharedMem['clientIDs']
        clientDict[f'{clientID}'] = addr[0]
        with Lock:
            sharedMem['clientIDs'] = clientDict
        # shared_memory[f'train_{cli}'] = True
        # sharedMem[f'{cli}_agg']       = False     # weight update flag of current client
        train = True
        # sharedMem[f'aggr'] = False
        # wait for weight updates from client untill training continues
        while train:
            # client commands to share weights after a round
            print('waiting for weights from client...')
            cmd_from_client = wait_for_acknowledge(cli, "sending weights after round.")
            if cmd_from_client=="sending weights after round.":
                print("Client weight sharing acknowledged...")
                cli.sendall(bytes("ACK", "utf-8"))
            print(f'{Style.BRIGHT}Aggregation server recieving checkpoint from {Fore.YELLOW}{addr} after round completion')
            updated_model = recieveData(cli)
            # cmd_from_client = wait_for_acknowledge(cli, "Training not completed")
            # if cmd_from_client!="Training not completed":
            #     train = False
            cli.sendall(bytes("wait for server update.", "utf-8"))
            # update client weights to shared memory
            with Lock:
                sharedMem[f'{clientID}_weights'] = copy.deepcopy(updated_model)
                # wait for aggregated model flag to be true
                sharedMem['aggr'] = True
            print('waiting for aggregation to be done...')
            while sharedMem['aggr']:
                # share weights with this client
                if sharedMem['serverUpdate']:
                    with Lock:
                        sharedMem['aggr'] = False
                    cli.sendall(bytes("sending updated best weights.", "utf-8"))
                    Ack_from_client = wait_for_acknowledge(cli, "ACK")
                    if Ack_from_client=="ACK":
                        updated_weights = copy.deepcopy(sharedMem['bestUpdate']).to('cpu')
                        print(f'{Style.BRIGHT}Sending checkpoint to {Fore.YELLOW}{addr} after aggregation to resume training...')
                        sendData(cli,updated_weights)
                    with Lock:
                        sharedMem['bestUpdate'  ] = False
                        sharedMem['serverUpdate'] = False
                    ack_from_client = wait_for_acknowledge(cli, "ACK")
                    if ack_from_client != "ACK":
                        raise ValueError('Client does not acknowledge command.')
                    finish = sharedMem['finish']
                    if finish:
                        train = False 
            if not train:
                break                
    print(f'{Style.BRIGHT} stoping client {addr}...')    
    
def main():
    args          = parse_args()
    args.dataset  = 'custom'    
    manager       = Manager()
    lock          =  manager.Lock()
    shared_memory = manager.dict()
    ars, _        = set4Train()
    with lock:
        shared_memory['totalRounds' ] = ars['rounds']
        shared_memory['numClients'  ] = 1
        shared_memory['bestUpdate'  ] = False
        shared_memory['AggModel'    ] = None
        shared_memory['bestModel'   ] = None
        shared_memory['clientIDs'   ] = {}
        shared_memory['aggr'        ] = False
        shared_memory['serverUpdate'] = False
        shared_memory['finish'      ] = False 
    
    aggProcess = Process(target=weightAgg, args=(shared_memory, args, lock))
    aggProcess.start()
    s = start_server(shared_memory['numClients'])
    
    processList = []
    id = 0
    while True:
        # connect with client
        client, address = s.accept()
        print(f"{Style.BRIGHT}Connection from {Fore.YELLOW}{address} has been established!")
        # clientList = shared_memory['clientIDs']
        # clientList.append(address)
        # shared_memory['clientIDs'] = clientList
        cliProcess = Process(target=handle, args=(client, address, shared_memory, args, id, lock))
        processList.append(cliProcess)
        cliProcess.start()
        id += 1
        time.sleep(1)
        # cliProcess.join()
        # finish = shared_memory['finish']
        if len(processList) == shared_memory['numClients']:
        # if len(processList)==shared_memory['numClients']:
            break
    
    run = True
    while run:
        while shared_memory['finish']:
            for cliProcess in processList:
                cliProcess.join()
            print(f'{Style.BRIGHT}{Fore.YELLOW}Distributed self-training finished, stopping aggregation server...')
            aggProcess.join()
            s.close()
            run = False
            break    

if __name__=="__main__":
    multiprocessing.set_start_method('forkserver')
    main()