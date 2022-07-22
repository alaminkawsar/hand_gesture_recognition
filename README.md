# hand_gesture_recognition

We build this model by the help of [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid). Firstly We use [ssd_mobile_net_v2_fpn_320x320](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) tensorflow model, and [coco-hand and tv hand-hand](https://www3.cs.stonybrook.edu/~cvl/projects/hand_det_attention/) data concurrently to train hand detection model. Then the detection hand cropped image is used to classify gestures using gesture classification model. 

