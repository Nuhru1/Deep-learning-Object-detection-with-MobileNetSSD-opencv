#import necessary packages
import cv2
import numpy as np


# path to the prototxt.txt file and caffe model.
PathPrototxt = " your path to the prototxt file"
PathCaffeModel = " your path to the caffe model"

# path to the image w want to detect objects on.
ImagePath = " your images path"


# initialization of list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Generate set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load our serialized model from disk
print("LOADING MODEL...")
net = cv2.dnn.readNetFromCaffe(PathPrototxt, PathCaffeModel)


#load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(ImagePath)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


#pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")

net.setInput(blob)   # our net can only be fed by the images blob, not by the image directly.
detections = net.forward() 


# ------------------loop over the detection-----------------------------------------------

for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > 0.6:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    

# show the output image
cv2.imshow("Output", image)
#cv2.imwrite(OutputPath, image)
cv2.waitKey(0)
