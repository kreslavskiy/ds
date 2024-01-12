import cv2

mode = int(input('Select picture: 1 - matchsticks, 2 - cookies, 3 - palm: '))
print(f'You selected picture {mode}')

if mode == 1:
  filename = 'matchstick.jpeg'
  min_threshold = 30
elif mode == 2:
  filename = 'cookies.jpeg'
  min_threshold = 70
elif mode == 3:
  filename = 'palm.jpeg'
  min_threshold = 17

image = cv2.imread(filename)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_image, min_threshold, 266, cv2.THRESH_BINARY_INV)
converted_image = cv2.bitwise_not(threshold)
contours, _ = cv2.findContours(converted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

total = 0
green_color = (0, 255, 0)
for contour in contours:
  area = cv2.contourArea(contour)
  if area > 100:
    total += 1
    cv2.drawContours(image, [contour], -1, green_color, 2)

cv2.imshow(f'Detected {total} items, - press any button to exit', image)
key  = cv2.waitKey(0)
if key != -1:
  cv2.destroyAllWindows()
