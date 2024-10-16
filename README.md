The module is meant to be used for a Distributed self-training with FPGA based accelerator

# Key Modules
1. **client_full.py** \
    Handles the client training process, weight sharing and re-training with updated weights from server. It should be run on the local PC (FPGA PC). Make sure to use correct IP address in the ```start_client()```,    
    to connect it with a local server, D4 board or the aggregation server. Current setting envoloves D4 board in the loop for data augmentation before training therefore it must be connected with D4 board. In        other scenario the D4 board can be skipped and client can be connected directly to the aggregation server.
    
3. **serverfull.py** \
    This is implimented for the test case scenarioon the local pc. This must be run on the D4 in actual case.
    
4. **aggregation_server.py** \
    This can be run on the local pc or on another pc for aggregating all the clients weights. Make sure to use correct server id on the client or D4 side to connect with the  aggregation server.

# Getting strated
-   Make correct environment and install the requirements
-   First run the the aggregation_server module by ensuring the correct setting (IP and port)
-   Run the serverfull if required
-   Run the client_full

# Data Handling
-   Check the ```argument()``` module in **server_full.py** for handling labelled, training and validation data. Current module assumes that the D4 should be the controller and will manage the data handling for self-training and validation. The code assumes the data directory paths to access the data and labels (only data in case of unlabeled data). Paths can be provided as inpute arguments while running the code. Use below snnipt to run the **serverfull.py** (replace all paths with your paths):

    ```
    python3 serverfull.py --unlabeled_dataPath dir/path --GT_dataPathTrain dir/path --GT_dataPathVal dir/path
    ```   
