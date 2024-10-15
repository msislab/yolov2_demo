import socket
import os
# import ast
import pickle
import struct
import zlib
from tqdm import tqdm


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
    # server_addr = ('10.125.150.41',2019)
    client.connect(server_addr)
    print(f"Connected to server!")
    return client

def recieveData(client):
    # size = int(wait_for_acknowledge(client,str(3)))
    size = struct.unpack('>I', client.recv(4))[0]
    client.sendall(bytes("ACK","utf-8"))
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

def main(client=None):
    # start client
    if client is None:
        client = start_client()
    # client.settimeout(15) #limit each communication time to 5s
    #listening to server command
    # print("Client is now waiting for server's command.")
    # client.settimeout(5) #limit each communication time to 5s
    #listening to server command
    print("Client is now waiting for server's command.")

    cmd_from_server = wait_for_acknowledge(client,"success...")
    if cmd_from_server == "success...":
        print('success...')
        client.sendall(bytes("ACK", "utf-8"))
    
    # client recieving necessary args to start training
    print('Recieving Distributed Self-Training setup parameters from aggregation Server...')
    cmd_from_server = wait_for_acknowledge(client, "Sending distributed training args.")
    if cmd_from_server=="Sending distributed training args.":
        client.sendall(bytes("ACK", "utf-8"))
    cmd_from_server = wait_for_acknowledge(client,"sending args dict.")
    if cmd_from_server=="sending args dict.":
        client.sendall(bytes("ACK", "utf-8"))
        trainArgs = recieveData(client)
        client.sendall(bytes("ACK", "utf-8"))
    cmd_from_server = wait_for_acknowledge(client, "sending starting checkpoint.")
    if cmd_from_server=="sending starting checkpoint.":
        client.sendall(bytes("ACK", "utf-8"))
        print('Recieving starting checkpoint from aggregation Server...')
        checkpoint = {}
        checkpoint["model"] = recieveData(client)
        client.sendall(bytes("ACK", "utf-8"))

    return checkpoint, trainArgs   

if __name__=="__main__":
    main()