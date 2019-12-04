import cv2
import numpy as np

def check_and_write_text(frame, pupil_consec, pupil_flag):
	if pupil_flag and pupil_consec >=6:
		cv2.putText(frame, 'PUPIL NOT DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

cap = cv2.VideoCapture('video3(1).mp4')
pupil_flag = False
pupil_consec = 0
while True:
	s, frame = cap.read()

	gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_f = cv2.GaussianBlur(gray_f, (9, 9), 0)
	_, threshold = cv2.threshold(gray_f, 30	, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# drawing circle!
	# contours_poly = [None]*len(contours)
	# centers = [None]*len(contours)
	# radius = [None]*len(contours)
	smax_contour = None
	if len(contours) > 2:
		contours_s = sorted(contours, key=lambda x: cv2.contourArea(x))
		smax_contour = cv2.contourArea(contours_s[-3])
		print(smax_contour, ' is the second max contour')
	for i, contour in enumerate(contours):
		# if cv2.contourArea(contour) >= cv2.contourArea(smax_contour):
		approx = cv2.approxPolyDP(contour, 3, True)
		area = cv2.contourArea(contour)
		if not smax_contour: smax_contour = 1500
		print('areaof frame', area, 'num lines : ', len(approx))
		if len(approx) > 7 and area >= 1500:

			peri = cv2.arcLength(contour, True)
			center, radius = cv2.minEnclosingCircle(approx)
			print(str((peri -(2*np.pi*radius))/peri))

			if (peri - 0.2*peri < 2*np.pi*radius < 0.2*peri + peri) :
				pupil_flag = False
				pupil_consec = 0
				cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
			else:
				pupil_consec += 1
				pupil_flag = True
				check_and_write_text(frame, pupil_consec, pupil_flag)
		else:
			pupil_consec += 1
			pupil_flag = True
			check_and_write_text(frame, pupil_consec, pupil_flag)

	# if pupil_flag and pupil_consec >=5:
	# 	cv2.putText(frame, 'PUPIL NOT DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
		# ctr = np.array(contour).reshape((-1,1,2)).astype(np.int32)
		# cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

	cv2.imshow("Threshold", threshold)
	# cv2.imshow("gray", gray_f)
	cv2.imshow("frame", frame)
	key = cv2.waitKey(30)
	if key == 27:
		break

cv2.distroyAllWindows()
