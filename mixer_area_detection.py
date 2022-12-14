import cv2
import numpy as np
import mediapipe as mp
from fps import FPS
from video_stream import VideoStream
import os

# Clean terminal
# os.system('clear')

print("Start Mixer Area Detection Program")

# Initialize the models needed from [mp.solutions]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

path = 'Mezcladora2'
screenSize = (854,480)

# Using either webcam or video
cap = cv2.VideoCapture(f'videos/{path}.MOV')
# cap = VideoStream(src=f'videos/{path}.MOV').start()
# cap = cv2.VideoCapture(0)

# Store a copy of the video
videoResult = cv2.VideoWriter(
    f'output_videos/{path}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15., screenSize)

# Start an instance to manage the video
fps = FPS().start()

# Initialize background substractor class with the number of Gaussian mixtures
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=10)
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Setting the morphological operations, giving the structuring element (shape)
# and the size
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Function for finding the human pose, returns a list with all the landmarks
# from the detected subject
def findHumanPose(image, results, draw=True):

    coordinates = []

    # Check for pose results
    if results.pose_landmarks:

        # If true, draw the landmarks on the main frame
        if(draw):
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w = image.shape

        # Get right hip coordinates according to the frame size
        right_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * w
        right_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * h

        # Get left hip coordinates according to the frame size
        left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * w
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h

        center_hip_x = (right_hip_x + left_hip_x)/2
        center_hip_y = (right_hip_y + left_hip_y)/2

        # print(f'Hip coordinates: ('f'{center_hip_x}, 'f'{center_hip_y})')

        coordinates = [center_hip_x, center_hip_y]

        # Iterates over the pose landmarks results to return a list
        # with the id of each landmark and the coordinates according
        # to the size of the image
        # for id, lm in enumerate(result.pose_landmarks.landmark):

        #     # Dimensions of the image
        #     h, w = image.shape

        #     # Coordinates of the landmark
        #     cx, cy = int(lm.x * w), int(lm.y * h)

        #     lmList.append([id, cx, cy])

        cv2.circle(frame, (int(center_hip_x), int(center_hip_y)),
                   10, (0, 200, 200), cv2.FILLED)

    return coordinates


# Start loop
while True:
    # Read the video source
    ret, frame = cap.read()

    # Breaks the loop if reading was not succesful
    if ret == False:
        break

    # Resizing frame for faster detection
    frame = cv2.resize(frame, screenSize)

    # Getting dimensions of the frame
    height, width, channels = frame.shape

    # Creating a frame with shades of gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Draw a black rectangle on frame to show the state
    # of the selected areas
    cv2.rectangle(frame, (0, 0), (int(width/2), 100), (0, 0, 0), -1)

    # Draw a black rectangle on frame to show the fps
    cv2.rectangle(frame, (int(width*.85), 0), (width, 70), (0, 0, 0), -1)

    # Setting main and secondary colors
    color = (150, 200, 0)
    sec_color = (150, 200, 0)

    # Initial state texts
    state_text = "Key Area empty"
    sec_state_text = "Sec Area empty"

    # Set the main area points acording the screen size
    main_max_w = width * 8/10
    main_min_w = width * 5/20
    main_max_h = height * 10/10
    main_min_h = height * 7/10

    # Store the main area points
    main_area_pts = np.array(
        [[int(main_max_w), int(main_max_h)],
         [int(main_min_w), int(main_max_h)],
         [int(main_min_w), int(main_min_h)],
         [int(main_max_w), int(main_min_h)]])

    # Set the secondary area points acording the screen size
    sec_max_w = width * 9/20
    sec_min_w = width * 0/20
    sec_max_h1 = height * 13.5/20
    sec_max_h2 = height * 10.5/20
    sec_min_h = height * 2.5/10
    # sec_max_w = width * 4/10
    # sec_min_w = width * 0/20
    # sec_max_h = height * 6/10
    # sec_min_h = height * 2.5/10

    # Store the secondary area points
    sec_area_pts = np.array(
        [[int(sec_max_w), int(sec_max_h2)],
         [int(sec_min_w), int(sec_max_h1)],
         [int(sec_min_w), int(sec_min_h)],
         [int(sec_max_w), int(sec_min_h)]])

    # Determine auxiliar images for defining the area where
    # the movement detector will work, setting the contourns
    # and the color
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [main_area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    sec_imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    sec_imAux = cv2.drawContours(sec_imAux, [sec_area_pts], -1, (100), -1)
    sec_image_area = cv2.bitwise_and(gray, gray, mask=sec_imAux)

    # Get the results of the Human Pose model
    # results = pose.process(frame)

    # Get the coordinates of the landmarks
    # coordinates = findHumanPose(image_area, results, draw=True)

    # Get the binary image areas and increase quality of the procces.
    # The white areas will represent movement.
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    sec_fgmask = fgbg.apply(sec_image_area)
    sec_fgmask = cv2.morphologyEx(sec_fgmask, cv2.MORPH_OPEN, kernel)
    sec_fgmask = cv2.dilate(sec_fgmask, None, iterations=2)

    # Find the contours on each mask
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    sec_cnts = cv2.findContours(sec_fgmask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    # For each contour, check if the area is higher than the
    # reference. The reference will change based on the movement
    # and area of interest.
    for cnt in cnts:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw a rectangle based on the coordinates of the contour
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Change the main state to indicate movement detection
            state_text = "Key Area Movement detected"
            # Change the main color to indicate the movement
            color = (255, 0, 255)

    for cnt in sec_cnts:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Change the secondary state to indicate movement detection
            sec_state_text = "Sec Area Movement detected"
            # Change the secondary color to indicate the movement
            sec_color = (255, 0, 255)

    # if len(coordinates) > 0:
    #     # print('Cx ' + str(coordinates[0])+ '  Cy ' + str(coordinates[1]))
    #     # print('X max ' + str(main_max_w)+ '  X min ' + str(main_min_w))
    #     # print('Y max ' + str(main_max_h)+ '  Y min ' + str(main_min_h))
    #     # Detects if a human is inside of the main area
    #     if coordinates[0] > main_min_w and coordinates[0] < main_max_w and coordinates[1] > main_min_h and coordinates[1] < main_max_h:
    #         state_text = 'Key Area Human detected'
    #         color = (200, 150, 0)
    #     # Detects if a human is inside of the secondary area
    #     if coordinates[0] > sec_min_w and coordinates[0] < sec_max_w and coordinates[1] > sec_min_h and coordinates[1] < sec_max_h:
    #         sec_state_text = 'Sec Area Human detected'
    #         sec_color = (200, 150, 0)

    # Draw the areas of interest
    cv2.drawContours(frame, [main_area_pts], -1, color, 2)
    cv2.drawContours(frame, [sec_area_pts], -1, sec_color, 2)

    # Set the state of the iteration
    cv2.putText(frame, state_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, sec_state_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, sec_color, 2)
    # Display the FPS indicator
    cv2.putText(frame, f'FPS:{int(fps.fps())}', (int(width*0.85), 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Write the output video
    videoResult.write(frame.astype('uint8'))

    # Show the secondary masl
    cv2.imshow('fgmask_sec', sec_fgmask)

    # Show the main mask
    cv2.imshow('fgmask', fgmask)

    # # Show the main frame
    cv2.imshow("frame", frame)

    # Wait for ['q'] key to close the loop
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    # Update the FPS counter
    fps.update()
# Stop the FPS
fps.stop()
# Print meant fps and duration at the end
print(" [INFO] duration: {:.2f}".format(fps.elapsed()))
print(" [INFO] mean fps: {:.2f}".format(fps.getMeanFps()))

# Close all windows
cap.release()
cv2.destroyAllWindows()
