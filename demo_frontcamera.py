"""
Documented on 03 Aug 2018
Qiu Biqing
GovTech Sensors and IoT
Adapted from code from: https://github.com/eldar/pose-tensorflow
and https://github.com/PJunhyuk/people-counting-pose
Model is pretrained coco resnet, did not do additional training
Have fun!
"""

# Import libraries from parent directory
from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from dataset.pose_dataset import data_to_input
from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut

# Import Python libraries
import numpy as np
import math
import cv2

# Function to rescale a frame; smaller frames are processed faster
# But model is less accurate on a smaller image
# Default rescale percentage is 50
def rescale(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Load the models and other parameters eg. joint positions to look for
cfg = load_config("pose_cfg_multi.yaml")
dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# 0 for front webcam, 1 for USB cam
cap = cv2.VideoCapture(0)

# To save the result video to file, uncomment the following:

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# width = int(cap.get(3))  # float
# height = int(cap.get(4)) # float
# out = cv2.VideoWriter('topview.avi', fourcc, 5.0, (int(width),int(height)))

while(True):
	# Reads a frame from camera
	ret, thisFrame = cap.read()

	# Convert frame to grayscale, rescale to 80%
	frame = rescale(np.stack((cv2.cvtColor(thisFrame,cv2.COLOR_BGR2GRAY),)*3, -1), 80)
	image_batch = data_to_input(frame)

	# Compute prediction with the CNN
	outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
	scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

	detections = extract_detections(cfg, scmap, locref, pairwise_diff)
	unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
	person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

	# Obtain an estimated number of people from the estimated number of points per person
	# Numbers are from code from: https://github.com/PJunhyuk/people-counting-pose
	point_num = 17
	people_num = int(person_conf_multi.size/(point_num*2))
	point_i = 0

	# Real number of people which we save
	people_real_num = 0

	# Iterate through the possible people 
	for people_i in range(0, people_num):
		point_count = 0
		for point_i in range(0, point_num):
			# If this point is not (0,0)
			if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1]) != 0:
				point_count = point_count +1
		# Set as actual person, if there are more than 5 points on that person
		if point_count>5:
			people_real_num= people_real_num + 1
			# Draw the online of red circles on screen
			for point_i in range(0, point_num):
				if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1])!= 0:
					cv2.circle(frame, (math.floor(person_conf_multi[people_i][point_i][0]),
										math.floor(person_conf_multi[people_i][point_i][1])),
										radius=5, color=(0, 0, 255), thickness=-1)
	# end of for
	# Write the number of people on screen
	cv2.putText(frame,'People No.: ' + str(people_real_num), 
	    (10, 50), 
	    cv2.FONT_HERSHEY_SIMPLEX,
	    1,
	    (0, 0, 255),
	    3, 
	    cv2.LINE_AA)
	# Show video
	cv2.imshow('frame', frame)

	# Uncomment the following to save video to file
	# out.write(frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
# Uncomment the following to save video to file
# out.release()
cv2.destroyAllWindows()

