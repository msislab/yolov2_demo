
def DS_configs():
    DEEPSORT = {'MODEL_TYPE': "resnet50",
    'MAX_DIST': 0.2, # The matching threshold. Samples with larger distance are considered an invalid match
    'MAX_IOU_DISTANCE': 0.7, # Gating threshold. Associations with cost larger than this value are disregarded.
    'MAX_AGE': 5, # Maximum number of missed misses before a track is deleted
    'N_INIT': 0, # Number of frames that a track remains in initialization phase
    'NN_BUDGET': 100} # Maximum size of the appearance descriptors gallery
    # print()
    return DEEPSORT

if __name__ == '__main__':
    configs = DS_configs()
    print()