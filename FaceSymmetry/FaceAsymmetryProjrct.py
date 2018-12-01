# USAGE
# python FaceAsymmetryProjrct.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import matplotlib.pyplot as plt
import numpy as np
import imutils
import dlib
import cv2


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

fa = FaceAligner(predictor, desiredFaceWidth=300)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('amineAF2.jpg')
image = imutils.resize(image, width=350)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
#cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=300)


    faceAligned = fa.align(image, gray, rect)


    grayFA = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

    rectsFA = detector(grayFA, 2)

for (i, rect) in enumerate(rectsFA):

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(grayFA, rect)
    shape = face_utils.shape_to_np(shape)
    print('the cordinates 2 :')
    print(shape)

    (x1, y1) = shape[27]
    (x2, y2) = shape[8]
    (x3, y3) = shape[30]

    halfFaceRight = faceAligned[0: 300, x3: 300]
    cv2.imshow("half Face right Befor", halfFaceRight)

    halfFaceLeft = faceAligned[0: 300, 0: x3]
    cv2.imshow("half Face left Befor", halfFaceLeft)

    halfCreated = cv2.flip(halfFaceLeft,1)

    leftFaceDuplicate = np.concatenate((halfFaceLeft, halfCreated), axis=1)
    cv2.imshow("duplicate face", leftFaceDuplicate)


    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(faceAligned, "Face #{} before".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(faceAligned, (x, y), 1, (0, 0, 255), -1)

    # draw the vertical symmetry line
    cv2.line(faceAligned, (x3, 0), (x3, 300), cv2.LINE_AA, 1)



    import uuid
    f = str(uuid.uuid4())
    cv2.imwrite("foo/" + f + ".png", faceAligned)

    # display the output images
    #cv2.imshow("Original", faceOrig)
    #cv2.imshow("Aligned", faceAligned)






    # load the input image, resize it, and convert it to grayscale
    image2 = cv2.imread('amine.jpg')
    image2 = imutils.resize(image2, width=350)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    #cv2.imshow("Input2", image2)
    rects2 = detector(gray2, 2)

    # loop over the face detections
    for rect in rects2:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig2 = imutils.resize(image2[y:y + h, x:x + w], width=300)

        faceAligned2 = fa.align(image2, gray2, rect)
        grayFA2 = cv2.cvtColor(faceAligned2, cv2.COLOR_BGR2GRAY)

        rectsFA2 = detector(grayFA2, 2)

        facesub = faceAligned2 - faceAligned
        #cv2.imshow("Face subtructed", facesub)

    for (i, rect) in enumerate(rectsFA2):

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(grayFA2, rect)
        shape = face_utils.shape_to_np(shape)
        print('the cordinates 2 :')
        print(shape)

        (x1, y1) = shape[27]
        (x2, y2) = shape[8]
        (x3, y3) = shape[30]

        halfFaceRight = faceAligned2[0: 300, x3: 300]
        cv2.imshow("half Face right After", halfFaceRight)

        halfFaceLeft = faceAligned2[0: 300, 0: x3]
        cv2.imshow("half Face left After", halfFaceLeft)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(faceAligned2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(faceAligned2, "Face #{} after".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(faceAligned2, (x, y), 1, (0, 0, 255), -1)


        # draw the vertical symmetry line
        cv2.line(faceAligned2, (x3, 0), (x3, 300), cv2.LINE_AA, 1)

        import uuid

        f = str(uuid.uuid4())
        cv2.imwrite("foo/" + f + ".png", faceAligned2)

        plt.figure(1)
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) ;plt.title("Input befor")
        plt.subplot(232)
        plt.imshow(cv2.cvtColor(faceOrig, cv2.COLOR_BGR2RGB));plt.title("InputFace befor")
        plt.subplot(233)
        plt.imshow(cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB));plt.title("Aligned Face befor")
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB));plt.title("Input after")
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(faceOrig2, cv2.COLOR_BGR2RGB));plt.title("InputFace after")
        plt.subplot(236)
        plt.imshow(cv2.cvtColor(faceAligned2, cv2.COLOR_BGR2RGB));plt.title("Aligned face after")
        plt.show()



    cv2.waitKey(0)