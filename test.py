from darkflow.net.build import TFNet
import cv2
cap = cv2.VideoCapture(0)

options = {"model": "cfg/tiny-yolo-voc.cfg", 
	"load": "bin/tiny-yolo-voc.weights", 
	"threshold": 0.2,
	"gpu":0
	 }

tfnet = TFNet(options)

while True:
        ret,frame = cap.read()
	result = tfnet.return_predict(frame)
	print(result)


