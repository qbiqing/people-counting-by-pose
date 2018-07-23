import os
import sys

import numpy as np
from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import random
import math
import cv2

cfg = load_config("pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	image_batch = data_to_input(frame)

	# Compute prediction with the CNN
	outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
	scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

	detections = extract_detections(cfg, scmap, locref, pairwise_diff)
	unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
	person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

	people_num = 0
	point_num = 17 # why?
	people_num = person_conf_multi.size/(point_num*2)
	people_num = int(people_num)
	point_i = 0

	people_real_num = 0

	for people_i in range(0, people_num):
		point_count = 0
		for point_i in range(0, point_num):
			if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1]) != 0:
				point_count = point_count +1
		if point_count>5: #more than 5 points, actual person
			people_real_num= people_real_num + 1
			for point_i in range(0, point_num):
				if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1])!= 0:
					cv2.circle(frame, (math.floor(person_conf_multi[people_i][point_i][0]),
														math.floor(person_conf_multi[people_i][point_i][1])),
											radius=5, color=(0, 0, 255), thickness=-1)
	# end of for
	cv2.putText(frame,'People No.: ' + str(people_real_num), 
	    (10, 50), 
	    cv2.FONT_HERSHEY_SIMPLEX,
	    1,
	    (255, 255, 255),
	    4, 
	    cv2.LINE_AA)
	# frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

