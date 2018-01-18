import cv2
import sys 

img = cv2.imread(sys.argv[1])
apple_haar = cv2.CascadeClassifier("./data/cascade.xml")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
apple = apple_haar.detectMultiScale(gray_img, 1.2, 3)
 
for apple_x,apple_y,apple_w,apple_h in apple:
    cv2.rectangle(img, (apple_x, apple_y), (apple_x+apple_w, apple_y+apple_h), (0,255,255), 2)
 
cv2.imshow('img', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
