import numpy as np
import cv2

# load image and shrink - it's massive
image = cv2.imread('/home/ayushi/pytorch-yolo-v3/pics/DSC_6195.jpg', 1)
'''
if image is not None:
    image = cv2.resize(image, None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
else:
    print("image not loaded")
    exit()

cv2.imshow('image', image)
'''
image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# filter out small lines between counties
kernel = np.ones((5,5),np.float32)/25
image_gray= cv2.filter2D(image_gray,-1,kernel)

# threshold the imageDSC_6194.JPGDSC_6194.JPG and extract contours
ret,thresh = cv2.threshold(image_gray,60,80,cv2.THRESH_BINARY_INV)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


# find the main island (biggest area)
cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

#create a  bounding rectangle
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
image = image[y:y+h, x:x+w]

cv2.imshow("Contour", image)

k = cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
