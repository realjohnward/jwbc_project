import numpy as np
import cv2 as cv
from visualize import visualize
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from time import sleep

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.


# STEP 5: Process the detection result. In this case, visualize it.
# image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image_copy, detection_result)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

    
while True:
    # sleep(0.)
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    cv.imwrite('./frame.png', frame)
    # Display the resulting frame
    

    image = mp.Image.create_from_file('./frame.png')
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    result = visualize(image_copy, detection_result)

    rgb_annotated_image = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    cv.imshow('frame',rgb_annotated_image)


    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()