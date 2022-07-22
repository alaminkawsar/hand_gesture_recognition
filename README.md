# hand_gesture_recognition

We build this model by the help of [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid). Firstly We use [ssd_mobile_net_v2_fpn_320x320](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) tensorflow model, [coco-hand and tv hand-hand](https://www3.cs.stonybrook.edu/~cvl/projects/hand_det_attention/) dataset concurrently to train hand detection model. Then the detection part  cropped image is used to classify gestures using gesture classification model. 

###Run this project:
1. Download Gesture Classification model([config file](https://github.com/hukenovs/hagrid))
2. Clone repository:
```bash
git clone https://github.com/alaminkawsar/hand_gesture_recognition
```

3. open this project in any ide (vscode,pycharm)

4. Change 'path_to_config' that is in [webcame_hand_gestures.py](https://github.com/alaminkawsar/hand_gesture_recognition/blob/master/classifier/webcam_hand_gestures.py) file.

5. Change checkpoint directory that is in [default.yaml](https://github.com/alaminkawsar/hand_gesture_recognition/blob/master/classifier/config/default.yaml)

```bash
experiment_name: MODEL_NAME

model:
    name: MODEL_NAME
```