import cv2

fist_cascade = cv2.CascadeClassifier('fist.xml')
if fist_cascade.empty():
  raise IOError('Unable to load the cascade classifier xml file')

webcam_video = cv2.VideoCapture(0)
scaling_factor = 0.375

key = -1
while key == -1:
  _, frame = webcam_video.read()
  frame = cv2.flip(frame, 1)
  frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  fist_rects = fist_cascade.detectMultiScale(gray_frame, 1.9, 16)

  green_color = (0, 255, 0)
  for (x, y, width, height) in fist_rects:
    cv2.rectangle(frame, (x,y), (x+width,y+height), green_color, 3)

  cv2.imshow('Detector', frame)

  key = cv2.waitKey(1)

webcam_video.release()
cv2.destroyAllWindows()
