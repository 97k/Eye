import cv2


def findThreshold(img):
    '''
    Finds an adaptive threshold based on the image. Threshold is chosen such that pupil of eye is visible and rest all is white (approx)
    Pupil as a blob is detected by the help of Simple Blob Detector algorithm.
    Threshold depends on image quality and many features which affects the image

    TODO: Add UI support to finetune thresholding.
    '''

    threshold = 20
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    # params.filterByCircularity = True
    # params.minCircularity = 0.4
    detector = cv2.SimpleBlobDetector_create(params)

    while threshold <= 60:
        threshold += 10
        _, thrash = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        keypoints = detector.detect(thrash)
        # print("keypoints", len(keypoints))

        if len(keypoints) >= 2:
            break

    else:
        return -1, False
    threshold = threshold - 10 if threshold >= 50 else threshold
    return threshold, True

 
def check_and_write_text(frame, pupil_consec, pupil_flag):
    if pupil_flag and pupil_consec >= 3:
        cv2.putText(frame, 'PUPIL NOT DETECTED', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        return 0, False
    return pupil_consec, pupil_flag
