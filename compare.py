
import cv2
import dlib
import time
import sys
import numpy as np

hogFaceDetector = dlib.get_frontal_face_detector()

dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
conf_threshold = 0.5

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
faceNeighbors = 5

def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
    frameOpenCVHaar = frame.copy()
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frameGray, 1.2, faceNeighbors)
    bboxes = []
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      round(frameHeight / 150), 4)
    return frameOpenCVHaar, bboxes



def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), round(frameHeight/150), 8)
    return frameOpencvDnn, bboxes


def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):

    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)
    
    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth
    
    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    faceRects = detector(frameDlibHogSmall, 0)
    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:

        cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),
                  int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), round(frameHeight/150), 4)
    return frameDlibHog, bboxes


def detectFaceDlibMMOD(detector, frame, inHeight=300, inWidth=0):

    frameDlibMMOD = frame.copy()
    frameHeight = frameDlibMMOD.shape[0]
    frameWidth = frameDlibMMOD.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)
    
    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth
    
    frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

    faceRects = detector(frameDlibMMODSmall, 0)
    
    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.rect.left()*scaleWidth), int(faceRect.rect.top()*scaleHeight),
                  int(faceRect.rect.right()*scaleWidth), int(faceRect.rect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), round(frameHeight/150), 4)
    return frameDlibMMOD, bboxes


source = 0
if len(sys.argv) > 1:
    source = argv[1]

cap = cv2.VideoCapture(source)
cv2.namedWindow("Face Detection Comparison", cv2.WINDOW_NORMAL)

while(1):
    hasFrame, frame = cap.read()

    t = time.time()
    outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
    tt_opencvHaar = time.time() - t
    fpsOpencvHaar = 1 / tt_opencvHaar
    label = "OpenCV Haar ; FPS : {:.2f}".format(fpsOpencvHaar)
    cv2.putText(outOpencvHaar, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    t = time.time()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
    tt_opencvDnn = time.time() - t
    fpsOpencvDnn = 1/tt_opencvDnn
    label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
    cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    t = time.time()
    outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
    tt_dlibHog = time.time() - t
    fpsDlibHog = 1/tt_dlibHog
    label = "DLIB HoG ; ; FPS : {:.2f}".format(fpsDlibHog)
    cv2.putText(outDlibHog, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    t = time.time()
    outDlibMMOD, bboxes = detectFaceDlibMMOD(dnnFaceDetector,frame)
    tt_dlibMmod = time.time() - t
    fpsDlibMmod = 1/tt_dlibMmod
    label = "DLIB MMOD ; FPS : {:.2f}".format(fpsDlibMmod)
    cv2.putText(outDlibMMOD, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    top = np.hstack([outOpencvHaar, outOpencvDnn])
    bottom = np.hstack([outDlibHog, outDlibMMOD])
    combined = np.vstack([top, bottom])
    cv2.imshow("Face Detection Comparison", combined)
    k = cv2.waitKey(10)
    if k == 27:
        break
cv2.destroyAllWindows()
