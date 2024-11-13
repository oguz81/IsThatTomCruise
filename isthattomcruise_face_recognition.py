# IS THAT TOM CRUISE
# This is the part of the "Is that Tom Cruise?" study.
# The face recognition model detects faces in video and
# decides whether the person is Tom Cruise or not.

# The model was also developed by Oguz Demirtas. It can be regarded
# as a kind of basic convolutional neural network study.
import os
import cv2
import sys
import time
import tensorflow as tf
from zipfile import ZipFile
from urllib.request import urlretrieve
from tensorflow.keras.preprocessing import image
import numpy as np

def whois(prediction):
  if prediction == '[1]':
    result = "Tom"
  else:
    result = "Who is?"
  return result

#URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
#asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")
#urlretrieve(URL, asset_zip_path)
#with ZipFile(asset_zip_path) as z:
  #z.extractall(os.path.split(asset_zip_path)[0])

#############################
# checks if there's a command-line command
# and then opens video capture source

s = 0
if len(sys.argv) > 1:
  s = sys.argv[1]
source = cv2.VideoCapture("#__path_to_video_here__#") #

#############################
win_name = "Face Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
output = "null"
# load the face detection model into the code
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
model = tf.keras.models.load_model('#__path_to_model_here__#')  # load the model

# specify the window size
in_width = 300
in_height = 300
mean = [104, 117, 123] # what kind of mean values are they?
conf_threshold = 0.7 # this is my man!

last_time = time.time()
while cv2.waitKey(1) != 27:
  has_frame, frame = source.read()
  if not has_frame:
    break
  #frame = cv2.flip(frame, 3) # 0 is upsidedown, 1 is mirror-like image
  frame_height = frame.shape[0]
  frame_width = frame.shape[1]

  # blob is the imput info which will sent into the neural network
  blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)

  net.setInput(blob) # blob is sent into the nn
  detections = net.forward() # detections is the output/prediction/result of the nn

  for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
      x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
      y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
      x_right_top = int(detections[0, 0, i, 5] * frame_width)
      y_right_top = int (detections[0, 0, i, 6] * frame_height)

      ss_width = x_right_top -x_left_bottom   # screenshot width
      ss_height = y_right_top - y_left_bottom # screenshot height

      # We need to make the height and the width of the screenshot equal.
      if ss_width > ss_height:
        add = (ss_width - ss_height) / 2
        y_right_top = y_right_top + int(add)
        y_left_bottom = y_left_bottom - int(add)
      if ss_height > ss_width:
        add = (ss_height - ss_width) / 2
        x_right_top = x_right_top + int(add)
        x_left_bottom = x_left_bottom - int(add)
      
      cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
      current_time = time.time()
      if current_time - last_time > 0.3: # take screenshot for every 0.3 seconds
        frameCopy = frame.copy()
        #frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
        name_is = "#__path_to_frame_saved__#" + str(current_time) + ".png"
        face = frameCopy[y_left_bottom:y_right_top, x_left_bottom:x_right_top] # problem is y_left_bottom
        my_image = cv2.imwrite(name_is, face)
        img =  image.load_img(name_is, target_size=(196, 196))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)
        output = str(predicted_class)
        last_time = current_time
      
      result = whois(output)
      label_size, base_line = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                           (x_left_bottom + label_size[0], y_left_bottom + base_line),
                           (255, 0, 0),
                            cv2.FILLED,)# this draws label rectangle
      cv2.putText(frame, result, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
  cv2.imshow(win_name, frame)
source.release()
cv2.destroyWindow(win_name)
