
import numpy as np

def returnIOU(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def iou2d(boxes1, boxes2):
    r1, c1 = boxes1.shape
    r2, c2 = boxes2.shape
    ious = np.zeros((r1,r2))
    for i in range(r1):
        box1 = boxes1[i]
        _ious = np.zeros(r2)
        for j in range(r2):
            box2 = boxes2[j]
            iou  = returnIOU(box1,box2)
            _ious[j] = iou
        ious[i] = _ious
    return ious         

if __name__ == '__main__':
    boxes1xyxy = np.array([[416, 218, 1695, 937]])
    boxes2xyxy = np.array([[219, 221, 1024, 940],
                           [0, 213, 35, 922],
                           [350, 188, 1447, 808]])
    IOUs = iou2d(boxes1xyxy, boxes2xyxy)

    print(IOUs)
    # array([[     0.4095,           0,     0.61196]])