import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame()

def findThreshold(img):
    threshold = 20
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    # params.filterByCircularity = True
    # params.minCircularity = 0.4
    detector = cv2.SimpleBlobDetector_create(params)

    while threshold <= 60:
        _, thrash = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        keypoints = detector.detect(thrash)
        # print("keypoints", len(keypoints))

        if len(keypoints) >= 2:
            break
        threshold += 10

    else:
        return -1, False
    threshold = threshold -10 if threshold >= 50 else threshold
    return threshold, True


def check_and_write_text(frame, pupil_consec, pupil_flag):
    if pupil_flag and pupil_consec >= 3:
        cv2.putText(frame, 'PUPIL NOT DETECTED', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        return 0, False
    return pupil_consec, pupil_flag


cap = cv2.VideoCapture('../video3(1).mp4')
pupil_flag = False
smax_contour = None
pupil_consec = 0
prev_frame = None
blinks = []
blink_counter = 0
window_sum = 0
window_count = 0
check_blink = False
nblinks = 0
blink_flag = False
threshold_flag = False
out = cv2.VideoWriter('result_vid3.avi', cv2.VideoWriter_fourcc(*'XVID'), 29.0, (int(cap.get(3)),int(cap.get(4))))

print('setting threshold...')
while True:
    s, frame = cap.read()
    if frame is None: break

    gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.GaussianBlur(gray_f, (5, 5), 0)

    #     # threshold_l, threshold_flag_l = findThreshold(left)
    #     # threshold_r, threshold_flag_r = findThreshold(right)
    threshold, threshold_flag = findThreshold(gray_f)
    if threshold_flag:
        print('Threshold setting done! \n threshold is ', threshold)
        break

while True:
    s, frame = cap.read()
    if frame is None:
        break

    gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.GaussianBlur(gray_f, (5, 5), 0)

    h, w, = gray_f.shape

    left = gray_f[:, 0:w // 2]
    right = gray_f[:, w // 2:w]

    _, threshold_img = cv2.threshold(gray_f, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # blink detection
    if prev_frame is not None:
        prev_frame_l, prev_frame_r = prev_frame[:, 0:w // 2], prev_frame[:, w // 2:w]

        score_l, _ = ssim(left, prev_frame_l, full=True)
        score_r, _ = ssim(right, prev_frame_r, full=True)

        window_count += 1

        # cv2.putText(frame, str(score_l), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        # cv2.putText(frame, str(score_r), (530, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        blinks.append((score_l, score_r))

        if score_l <= 0.85 and score_r <= 0.85:
            blink_counter += 1
        blink_flag = True if 2 <= blink_counter < 10 else False
        # blink_flag = True
        if window_count > 12 or blink_counter > 12:
            window_count = 0
            blink_counter = 0

        if window_count == 12 and window_count - blink_counter <= 5:
            nblinks += 1
            # check_blink = True
            # blink_flag = False
            print('blink detected, logging...', str(window_count), str(blink_counter))

        elif window_count == 12 and window_count - blink_counter > 5:
            window_count = 0
            blink_counter = 0

        cv2.putText(frame, 'Blinks: ' + str(nblinks), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    # if len(contours) > 2:
    #     contours_s = sorted(contours, key=lambda x: cv2.contourArea(x))
    #     smax_contour = cv2.contourArea(contours_s[-3])
    # # print(smax_contour, ' is the second max contour')
    if len(contours) == 0:
        pupil_consec, pupil_flag = check_and_write_text(frame, pupil_consec + 1, True)

    if pupil_flag:
        pupil_consec += 1
    pupil_consec, pupil_flag = check_and_write_text(frame, pupil_consec, pupil_flag)

    pupil_temp = 0

    for i, contour in enumerate(contours):
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
        approx = cv2.approxPolyDP(contour, 3, True)
        area = cv2.contourArea(contour)
        # if not smax_contour: smax_contour = 2000
        # print('area of frame', area, 'num lines : ', len(approx))
        if len(approx) >= 6 and 2000 <= area:

            peri = cv2.arcLength(contour, True)
            center, radius = cv2.minEnclosingCircle(approx)
            # print('peri difference', str((peri - (2 * np.pi * radius)) / peri))

            if peri - 0.20 * peri <= 2 * np.pi * radius <= 0.1 * peri + peri:
                pupil_flag = False
                pupil_consec = 0
                pupil_temp = 0
                cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
            else:
                pupil_temp += 1

        else:
            # print('In else of contour plotting, ', area, len(approx))
            # if len(approx) < 6:
            pupil_temp += 1
            #     pupil_flag = True
            #     pupil_consec, pupil_flag = check_and_write_text(frame, pupil_consec, pupil_flag)
    if pupil_temp >= len(contours) - 3:
        pupil_consec, pupil_flag = check_and_write_text(frame, pupil_consec + 1, True)

    out.write(frame)

    if not blink_flag:
        prev_frame = gray_f

    # cv2.imshow("Threshold_l", threshed_img_l)
    # cv2.imshow("Threshold_r", threshed_img_r)
    cv2.imshow("Threshold", threshold_img)
    # cv2.imshow("gray", gray_f)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

df['vid_l'] = [x[0] for x in blinks]
df['vid_r'] = [x[1] for x in blinks]

df.to_csv('debug.csv', index=False)
plt.plot(df['vid_l'])
plt.plot(df['vid_r'])
plt.show()
cap.release()
out.release()
print('Done')
cv2.destroyAllWindows()
