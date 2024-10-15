import numpy as np
import torch
import sys
from os.path import exists as file_exists, join
# from os.path import exists as join
import warnings
warnings.filterwarnings("ignore")
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
# from .deep.reid_model_factory import show_downloadeable_models, get_model_link, is_model_in_factory, \
#     is_model_type_in_model_path, get_model_type, show_supported_models
from .deep.reid_model_factory import is_model_in_factory, \
    is_model_type_in_model_path, get_model_type

sys.path.append('deep_sort/deep/reid')
from torchreid.utils import FeatureExtractor
import cv2
# from torchreid.utils.tools import download_url
# import gdown

# show_downloadeable_models()

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        # models trained on: market1501, dukemtmcreid and msmt17
        if is_model_in_factory(model):
            # download the model
            model_path = join('/home/zafar/yolov2_demo/deep_sort/deep/checkpoint', model + '.pth')
            # if not file_exists(model_path):
            #     gdown.download(get_model_link(model), model_path, quiet=False)

            self.extractor = FeatureExtractor(
                # get rid of dataset information DeepSort model name
                model_name=model.rsplit('_', 1)[:-1][0],
                model_path=model_path,
                device=str(device),
                # verbose=False
            )
        else:
            if is_model_type_in_model_path(model):
                model_name = get_model_type(model)
                self.extractor = FeatureExtractor(
                    model_name=model_name,
                    model_path=model,
                    device=str(device)
                )
            else:
                print('Cannot infere model name from provided DeepSort path, should be one of the following:')
                # show_supported_models()
                exit()
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "euclidean", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, hc_boxes_norm, use_yolo_preds=True):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, hc_boxes_norm)

        # output bbox identities
        trks_confirm, trk_missed, trk_new = [], [], []
        trks_confirm_n, trk_missed_n, trk_new_n = [], [], []
        for tracks_iter_id,track in enumerate(self.tracker.tracks):
            if track.is_deleted():
	            # if not track.is_confirmed()  or track.time_since_update > 1:
                continue

            
            # if track.is_confirmed()  or track.time_since_update > 1:
            if track.is_confirmed()  and track.time_since_update < 1:
                conf = track.get_yolo_pred().confidence
                if use_yolo_preds:
                    det = track.get_yolo_pred()
                    x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
                else:
                    box = track.to_tlwh()
                    x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                class_id = track.class_id
                trks_confirm.append(np.array([x1, y1, x2, y2, conf*10000, class_id, track_id, tracks_iter_id], dtype=np.int))
                trks_confirm_n.append(track.get_org_input())                        
            
            if track.is_confirmed()  and track.time_since_update >= 1:
                conf = track.get_yolo_pred().confidence
                # if use_yolo_preds:
                #     det = track.get_yolo_pred()
                #     x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
                # else:
                #     box = track.to_tlwh()
                #     x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                
                track_id = track.track_id
                class_id = track.class_id
                minS = 15
                if box[2]>minS and box[3]>minS:
                    trk_missed.append(np.array([x1, y1, x2, y2, conf*10000, class_id, track_id, tracks_iter_id], dtype=np.int))
                    trk_missed_n.append(track.get_org_input())
                
            if track.is_tentative():
                conf = track.get_yolo_pred().confidence
                if use_yolo_preds:
                    det = track.get_yolo_pred()
                    x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
                else:
                    box = track.to_tlwh()
                    x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                class_id = track.class_id
                trk_new.append(np.array([x1, y1, x2, y2, conf*10000, class_id, track_id, tracks_iter_id], dtype=np.int))
                trk_new_n.append(track.get_org_input())
                
                
        if len(trks_confirm) > 0:   trks_confirm = np.stack(trks_confirm, axis=0)
        if len(trk_missed) > 0:     trk_missed = np.stack(trk_missed, axis=0)
        if len(trk_new) > 0:        trk_new = np.stack(trk_new, axis=0)
            
        if len(trks_confirm_n) > 0:   trks_confirm_n = np.stack(trks_confirm_n, axis=0)
        if len(trk_missed_n) > 0:     trk_missed_n = np.stack(trk_missed_n, axis=0)
        if len(trk_new_n) > 0:        trk_new_n = np.stack(trk_new_n, axis=0)
            
            
        return trks_confirm, trk_missed, trk_new, trks_confirm_n,trk_missed_n,trk_new_n    
    
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        # image_size=(128, 256)
        # pixel_mean=[0.485, 0.456, 0.406],
        # pixel_std=[0.229, 0.224, 0.225],

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]      # im --> h,w,c
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (128, 256))
            im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            im = (im - mean) / std
            im_crops.append(im)
        _im_crops = torch.stack(im_crops, dim=0)
        if im_crops:
            features = self.extractor(_im_crops)
        else:
            features = np.array([])
        return features
