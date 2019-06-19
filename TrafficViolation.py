import datetime
import winsound
import cv2
import imutils

firstFrame = None
light = "Red"
cnt = 0
dynamic = False
min_area = 500
duration = 200  # millisecond
freq = 900  # Hz

# cam specific properties
zone1 = (100, 150)
zone2 = (450, 145)
thres = 30
dynamic = False

text = ""
isCar = False
cropped_cars = []  # list for taking all violated car's snap

cap = cv2.VideoCapture('video.mp4')
while True: 
    # reads frames from a video 
    ret, frame = cap.read()
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, thres, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        if (zone1[0] < (x + w / 2) < zone2[0] and (y + h / 2) < zone1[1] + 100 and (
                y + h / 2) > zone2[1] - 100):
            isCar = True

        if light == "Red" and zone1[0] < (x + w / 2) < zone2[0] and zone1[1] > (y + h / 2) > \
                zone2[1]:
            winsound.Beep(freq, duration)
            rcar = frame[y:y + h, x:x + w]
            rcar = cv2.resize(rcar, (0, 0), fx=4, fy=4)
            cropped_cars.append(rcar)
            cv2.imwrite('Reported_car/' + str(cnt) + ".jpg", rcar)
            cv2.imshow("Reported", rcar)
            cnt += 1
            text = "<Violation>"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if dynamic or not isCar:
        firstFrame = gray
    # draw the text and timestamp on the frame
    if light == "Green":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, zone1, zone2, (255, 0, 0), 2)
    cv2.putText(frame, "Signal Status: {}".format(light), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, "{}".format(text), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    try:
        cv2.imshow('frame', frame)
        cv2.imshow('list_of_cars', cropped_cars)
    except:
        pass
    pack = {'cnt': cnt}
      
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
cv2.destroyAllWindows() 
