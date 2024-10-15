# import bbox as BOX
import torch
import numpy as np


def xywh2xyxy(x):
    '''
    converts boxes from xywh to xyxy format \n
    x : array of boxes --> shape of x is (n,4)
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    for i in range(x.shape[0]):
        _x = x[i]
        _y = np.zeros(4)
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        _y[0] = _x[0] # x1
        _y[1] = _x[1] # y1
        _y[2] = _x[0] + _x[2]  # x2
        _y[3] = _x[1] + _x[3]  # y2
        y[i]  = _y        
    return y

def xyxy2xywh_1(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] # top left x
    y[1] = x[1] # top left y
    y[2] = x[2] - x[0]  # bottom right x
    y[3] = x[3] - x[1]  # bottom right y
    return y

def xyxy2xywh_center(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[2] + x[0])/2  # bottom right x
    y[1] = (x[3] + x[1])/2  # bottom right y
    y[2] = x[2] - x[0]  # bottom right x
    y[3] = x[3] - x[1]  # bottom right y
    return y


# def xywh2xywh_shoaib0(x):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = x[:, 0]   # x left
#     y[:, 1] = x[:, 1]   # y top
#     y[:, 2] = x[:, 2]    # width
#     y[:, 3] = x[:, 3]    # height
#     return y

# def xywh2xywh_shoaib2(x):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = x[:, 0]+2  # x left
#     y[:, 1] = x[:, 1]+2  # y top
#     y[:, 2] = x[:, 2]-4   # width
#     y[:, 3] = x[:, 3]-4   # height
#     return y

# def xywh2xywh_shoaib4(x):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = x[:, 0]+4  # x left
#     y[:, 1] = x[:, 1]+4  # y top
#     y[:, 2] = x[:, 2]-8   # width
#     y[:, 3] = x[:, 3]-8   # height
#     return y


def returnIOU(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # iou = interArea / float(boxAArea + boxBArea - interArea)
    return interArea / float(boxAArea + boxBArea - interArea)

def iou2d(boxes1, boxes2):
    r1, _ = boxes1.shape
    r2, _ = boxes2.shape
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

# def annotate_boxes(annotator, det, names, istensor=False, color=(255,0,0), pad=0):
#     if istensor: det = det.cpu().detach().numpy()
#     if len(det[0])>6:
#         boxes, confs, classes, tid = det[:, 0:4], det[:,4], det[:,5], det[:,6]
#         for j, (box, conf, cls, trk) in enumerate(zip(boxes, confs, classes, tid)):
#             label = f'{names[int(cls)]} conf:{conf/10000:.2f} track:{trk}'
#             box[0],box[1],box[2],box[3] = box[0]-pad,box[1]-pad,box[2]+pad*2,box[3]+pad*2
#             annotator.box_label(box, label, color)
#     else:
#         boxes, confs, classes = det[:, 0:4], det[:,4], det[:,5]
#         for j, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
#             box[0],box[1],box[2],box[3] = box[0]-pad,box[1]-pad,box[2]+pad*2,box[3]+pad*2
#             label = f'{names[int(cls)]} conf:{conf:.2f}'
#             annotator.box_label(box, label, color)
            
            
def match_boxes(boxes1, boxes1_n, boxes2, boxes2_n, iou_thresh = 0.80):

    # if torch.is_tensor(boxes1): boxes1 = boxes1.cpu().detach().numpy()
    # if torch.is_tensor(boxes2): boxes2 = boxes2.cpu().detach().numpy()
    
    # boxes_boxes1 = BOX.BBox2DList(boxes1[:,:4])       # X1, Y1, X2, Y2, Conf, Cls, TID
    # boxes_boxes2 = BOX.BBox2DList(boxes2[:,:4])       # X1, Y1, X2, Y2, Conf, Cls, TID
    
    boxes_boxes1 = boxes1[:,:4]       # X, Y, W, H
    boxes_boxes2 = boxes2[:,:4]       # X, Y, W, H
    boxes_boxes1XYXY = xywh2xyxy(boxes_boxes1)       # X1, Y1, X2, Y2
    boxes_boxes2XYXY = xywh2xyxy(boxes_boxes2)       # X1, Y1, X2, Y2

    # conf_boxes1  = boxes1[:,4]/10000
    # conf_boxes2  = boxes2[:,4]
    
    # class_boxes1 = boxes1[:,5]
    # class_boxes2 = boxes2[:,5]

    matched_idx = []
    matched1, unmatched1 = [], []
    matched2, unmatched2 = [], []
    matched1_n, unmatched1_n = [], []
    matched2_n, unmatched2_n = [], []
    
    # IoUs = BOX.metrics.multi_iou_2d(boxes_boxes1, boxes_boxes2)
    IoUs = iou2d(boxes_boxes1XYXY, boxes_boxes2XYXY)
    iou = np.logical_and(IoUs>iou_thresh,IoUs<1)
    
    len_box1,len_box2 = iou.shape
    
    for i in range(len_box1):
        # iiou = iou[i] if iou.shape[1]==1 else iou[:,i]
        # if iiou.any():
        if iou[i].any():
            matched1.append(boxes1[i])
            matched1_n.append(boxes1_n[i])
        else:
            unmatched1.append(boxes1[i])
            unmatched1_n.append(boxes1_n[i])
    
    for j in range(len_box2):
        # iiou = iou[j] if iou.shape[1]==1 else iou[:,j]
        # if iiou.any():
        if iou[:,j].any():
            matched2.append(boxes2[j])
            matched2_n.append(boxes2_n[j])
        else:
            unmatched2.append(boxes2[j])
            unmatched2_n.append(boxes2_n[j])
    
            
            # for j in range(len_box2):
            #     if iiiou: 
            #         matched_idx.append([i,j])
            # for j,iiiou in enumerate(iiou):
            #     if not iiiou:
	    
    
    return np.array(matched1), np.array(unmatched1), np.array(matched2), np.array(unmatched2), np.array(matched1_n), np.array(unmatched1_n), np.array(matched2_n), np.array(unmatched2_n), np.array(matched_idx)
    
    # if len(iou)==1:
    #     for j in range(len(iou)):
    #         if iou[0][j]:
    #             matched_idx.append([0,j])
    #             matched1.append(boxes1)
    #             matched2.append(boxes2[j])
    #         else:
    #             unmatched1.append(boxes1)
    #             unmatched2.append(boxes2[j])
    #     if not matched1==[]: matched1 = matched1[0]
    #     if not unmatched1==[]: unmatched1 = unmatched1[0]
    #     if not matched2==[]: matched2 = matched2[0]
    #     if not unmatched2==[]: unmatched2 = unmatched2[0]
        
    # else:       
    #     for i in range(iou.shape[0]):
    #         for j in range(iou.shape[1]):
    #             if iou[i,j]:
    #                 matched_idx.append([i,j])
    #                 matched1.append(boxes1[i])
    #                 matched2.append(boxes2[j])
    #             else:
    #                 unmatched1.append(boxes1[i])
    #                 unmatched2.append(boxes2[j])
                
def update_missed_box(deepsort,matched_trk_missed, matched_trk_missed_n, matched_det_yolo_lc, matched_det_yolo_lc_n, im0, count):
    trker_ = deepsort.tracker
    for i in range(len(matched_det_yolo_lc)):
        trk_ = matched_trk_missed
        det_ = matched_det_yolo_lc
        box_ = det_[0][:4]
        box_xywh = xyxy2xywh_1(box_)
        conf = det_[0][4]
        cls_ = det_[0][5]
        ttid_ = trk_[0][7]
        track_ = trker_.tracks[ttid_]
        kf_ = trker_.kf
        features = deepsort._get_features([box_],im0)
        det__ = Detection_(box_xywh, conf, features[0])
        track_.update(kf_,det__,cls_, matched_det_yolo_lc_n[i])
        count+=1
    
    return deepsort, count

def save_results(boxes,boxes_n,txt_path,im):
    hh, ww, _ = im.shape
    f = open(txt_path, 'a')
    if len(boxes) > 0:
        if len(boxes_n) > 0:
            for j, (box,box_n) in enumerate(zip(boxes,boxes_n)):
                l,t,w,h,conf,cls = box_n
                f.write(('%g ' * 6  + '\n') % (cls, l, t, w, h, conf))  # to YOLO format                        
                # f.write(('%g ' * 5  + '\n') % (cls, l, t, w, h))  # to YOLO format                        
        else:
            for j, box in enumerate(boxes):
                xywh = xyxy2xywh_center(box[0:4])
                bl,bt,bw,bh = xywh[0]/ww,xywh[1]/hh,xywh[2]/ww,xywh[3]/hh
                cls = int(box[5])
                conf = float(box[4]/10000)
                f.write(('%g ' * 6  + '\n') % (cls, bl, bt, bw, bh, conf))  # to YOLO format                        
                # f.write(('%g ' * 5  + '\n') % (cls, bl, bt, bw, bh))  # to YOLO format                        
            
                             
class Detection_(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature.cpu(), dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
