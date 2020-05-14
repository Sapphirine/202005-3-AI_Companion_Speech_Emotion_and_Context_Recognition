import os
import numpy as np 
import cv2
import dlib
from collections import deque
import time 



def get_live_video(queue,e):
	#emotion = "neutral"
	alt_frame = True
	video = cv2.VideoCapture(0)
	detector = dlib.get_frontal_face_detector()
	count = 1
	while True:
# 		print("here")
		(grabbed,frame) = video.read()
		if not grabbed :
			print("not grabbed")
			break
		if alt_frame:
			faces = detector(frame)
			frame_copy = frame.copy()

			for face in faces:
# 				print("herefaces")
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()
				face = frame[y1:y2, x1:x2]

				cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(255,0,0),thickness=7)
				#cv2.putText(frame_copy, predicted_emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
				# if not predicted_emotion.empty():
				# 	emotion = predicted_emotion.get()
				# cv2.putText(frame_copy, emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
# 				print("hereframe")
				cv2.imshow('Frame',frame_copy)

				frame = cv2.resize(face, (224, 224)).astype("float32")
				norm_image = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				
				e.wait()
				if e.is_set():
					#print("pushing to queue")
					queue.put(norm_image)

				count += 1
		alt_frame = not alt_frame
			
		if cv2.waitKey(10) == ord('q') : #wait until keyboard interrupt
			break
	cv2.destoryAllWindows()
 			