## Yolov2 Pytorch Implementation

This repository aims to learn and understand the YOLO algorithm. I am a beginner of deep learning, and I found the best way to learn a deep learning algorithm is to implement it from scratch. So if you also feel this way, just follow this repo! The code in this projects is clear and easier to understand, and I also documented it as much as possible. 

<div style="color:#0000FF" align="center">
<img src="images/result3.png" width="350"/>
<img src="images/result1.png" width="350"/> 
</div>

## Purpose

- [x] train pascal voc
- [x] multi-GPUs support
- [x] test
- [x] pascal voc validation
- [x] data augmentation
- [x] pretrained network
- [x] reorg layer
- [x] multi-scale training
- [ ] reproduce original paper's mAP

## Main Results

| | training set | test set | mAP@416 | mAP@544 |
| :--: | :--: | :--: | :--: | :--: |
|this repo|VOC2007+2012|VOC2007|72.7|74.6|
|original paper|VOC2007+2012|VOC2007|76.8|78.6|

Running time: ~19ms (52FPS) on GTX 1080


## Prerequisites
- python 3.5.x
- pytorch 0.4.1
- tensorboardX
- opencv3
- pillow

## Preparation

First clone the code

    git clone https://github.com/Zafar343/yolov2-pytorch.git
    
Install dependencies

	pip install -r requirements.txt

Then create some folder

    mkdir output 
    mkdir data
    mkdir results

Download the pretrained weights

```
wget http://pjreddie.com/media/files/yolo-voc.weights
```

       
### Inference & Visualization

	python demo.py
 
### Download pretrained network

    cd yolov2.pytorch
    cd data
    mkdir pretrained
    cd pretrained
    wget https://pjreddie.com/media/files/darknet19_448.weights
















