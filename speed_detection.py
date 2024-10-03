import cv2
import dlib
import time
import math

# algo=cv2.bgsegm.createBackgroundSubtractorMOG2()
carCascade = cv2.CascadeClassifier('vech.xml')
video = cv2.VideoCapture("cars.mp4")
# video = cv2.VideoCapture("cars.mp4")

WIDTH = 1280 #screen size
HEIGHT = 720 #screen size

def estimateSpeed(location1, location2):  #mathematics and physics work
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2)) #to find distance
    # ppm = location2[2] / carWidht
    ppm = 8.8  #pixel per minute default for monitoring in a video range upto 3km
    d_meters = d_pixels / ppm   #distance/pixels per minute 
    fps = 18  #default frames per seconds for capturing frames 
    speed = d_meters * fps * 3.6 #1 hour has 3600 seconds
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {} #storing car data in dictionaries or objects 
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outtraffic.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))
    #reading and converting video into mjpg file
    while True:
        start_time = time.time() #start timer
        rc, image = video.read()  #reading a video
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))  #resizing the image
        resultImage = image.copy() 

        frameCounter = frameCounter + 1 #framecounting to coount frames
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:  #checking thresholding.
                carIDtoDelete.append(carID) #appending our data less than 7

        
        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from list of trackers. ')
            print("Removing carID " + str(carID) + ' previous location. ')
            print("Removing carID " + str(carID) + ' current location. ')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        
        if not (frameCounter % 10): #not equals to zero means its not true
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #thresholding
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24)) #erosion of dilation image

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4) #feature extraction

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    
    cv2.destroyAllWindows()
    out.release()
#here the demo has been runn from here the main window willbw started 
# for(_x, _y, _w, _h) in cars:
#                 x = int(_x)
#                 y = int(_y)
#                 w = int(_w)
#                 h = int(_h)

#                 x_bar = x + 0.5 * w
#                 y_bar = y + 0.5 * h
# while(True):
#     if(cv2.VideoCapture()!=0):
#         print("hello there is a n error ")
#     elif(x[2]>23 and  x[-1]<58):
#         length=_x+_y+_w+_x
#         print(length)
#         print("hence the lenghth of a car is not satisfying in our pixels ")
#         height=length/2
#         breadth=x_bar+y_bar*(x_bar*y_bar)%100
#     else:
#         print("the vide can be run suucessfully ")
#     if(fps/2==0):
#         breadth*height*length
#         cv2.imshow("result",resultImage)
if __name__ == '__main__':
    trackMultipleObjects()


 
    
