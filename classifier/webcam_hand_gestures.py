import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import time
import tensorflow as tf
import cv2
import numpy as np

from util import label_map_util


'''
run.py
'''
import logging
import argparse
import torch.utils
import torch.optim

import torch

from torch import Tensor
from PIL import Image, ImageOps
from typing import Optional, Tuple
from torchvision.transforms import functional as f


from preprocess import get_transform
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from utils import set_random_state, build_model, collate_fn


#Classification Config file
path_to_config = '/home/kawsar/Desktop/Deep Learning/HandGesture/Research_Hagrid_Model/hagrid/classifier/config/default.yaml'


# Only tested for tensorflow 2.4.1, opencv 4.5.1

#Hand Detection Model information
MODEL_NAME = 'ssd_mobilenet_v2_fpn_320'
MODEL_DATA_COCO_HAND = 'model_data_coco_hand'
PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), MODEL_DATA_COCO_HAND, MODEL_NAME, 'saved_model')
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_DATA_COCO_HAND, MODEL_NAME, 'label_map.pbtxt')


logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

targets = {
    1: "call",
    2: "dislike",
    3: "fist",
    4: "four",
    5: "like",
    6: "mute",
    7: "ok",
    8: "one",
    9: "palm",
    10: "peace",
    11: "rock",
    12: "stop",
    13: "stop inverted",
    14: "three",
    15: "two up",
    16: "two up inverted",
    17: "three2",
    18: "peace inverted",
    19: "no gesture"
}


class Demo:
    
    def __init__(self):
        self.sum = 0
        
    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((224, 224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)

    
    def run(classify, frame, num_hands: int = 2, threshold: float = 0.5) -> None:
        
        processed_frame, size, padded_size = Demo.preprocess(frame)
        pred = classify(processed_frame)['gesture']
        index = pred.argmax()+1
        
        return targets[int(index)]
        
    
    def load_model(path_to_config):

        conf = OmegaConf.load(path_to_config)
        
        set_random_state(conf.random_state)

        num_classes = len(conf.dataset.targets)
        conf.num_classes = {"gesture": num_classes, "leading_hand": 2}

        model = build_model(
            model_name=conf.model.name,
            num_classes=num_classes,
            checkpoint=conf.model.get("checkpoint", None),
            device=conf.device,
            pretrained=conf.model.pretrained,
            freezed=conf.model.freezed
        )
        
        model.eval();
        
        return model
    




# Load label map and obtain class names and ids
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util.create_category_index(
    label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=1, use_display_name=True
    )
)

def visualise_on_image(classify_model, image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            epx = 10;
            epy = 20;
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            
            
            cropped_image = image[ymin-epy:ymax+epy+5,xmin-epx:xmax+epx]
            gesture = "NO Gesatu"
            if(ymax-ymin>=0 and xmax-xmin>=0):
                gesture = Demo.run(classify_model,cropped_image)
                print(gesture)
                #return cropped_image
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, f"{gesture}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image


if __name__ == '__main__':
    
    # Load the model
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")
    
    
    classify_model = Demo.load_model(path_to_config=path_to_config)
    # print(classify_model)
    
    # Open Video Capture (Camera)
    video_capture = cv2.VideoCapture(0)
    tic = time.time()

    while True:
      ret, frame = video_capture.read()
      if not ret:
          print('Error reading frame from camera. Exiting ...')
          break
    
      frame = cv2.flip(frame, 1)
      image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      # The model expects a batch of images, so also add an axis with `tf.newaxis`.
      t1 = time.time()
      input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

      # Pass frame through detector
      detections = detect_fn(input_tensor)

      # Detection parameters
      score_thresh = 0.4
      max_detections = 4

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      scores = detections['detection_scores'][0, :max_detections].numpy()
      bboxes = detections['detection_boxes'][0, :max_detections].numpy()
      labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      labels = [category_index[n]['name'] for n in labels]

      # Display detections
      frame=visualise_on_image(classify_model, frame, bboxes, labels, scores, score_thresh)
      t2 = time.time()
      #print("Time to give output: ",t2-t1)

      toc = time.time()
      fps = int(1/(toc - tic))
      tic = toc
      cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      cv2.imshow("Hand theremin", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Exiting ...")
    video_capture.release()
    cv2.destroyAllWindows()