import cv2
import numpy as np
import imutils


image = cv2.imread('pos_7.jpg')
template = cv2.imread('template_new.jpg')

# convert the image to greyscale

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Apply Gauss Filter on the image

gauss_image = cv2.GaussianBlur(imageGray,(5,5),0)
# Apply adaptive _threshold to the image and template

adaptive_threshold_image = cv2.adaptiveThreshold(gauss_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
adaptive_threshold_template = cv2.adaptiveThreshold(templateGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# Template Matching

result = cv2.matchTemplate(adaptive_threshold_image,adaptive_threshold_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h,w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)


cv2.imwrite('Cursor_Identication_1.jpg',image)
cv2.imshow('gauss_image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()














































'''import numpy as np
import argparse
import imutils
import glob
import cv2


import numpy as np
import cv2
 
image = cv2.imread('pos_2.jpg')
template = cv2.imread('template.png')
 
# resize images
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
#template = cv2.resize(template, (0,0), f) 
 
# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
# Find template
result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = min_loc
h,w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)
 
# Show result
cv2.imshow("Template", template)
cv2.imshow("Result", image)
 
cv2.moveWindow("Template", 10, 50)
cv2.moveWindow("Result", 150, 50)
 
cv2.waitKey(0)'''

'''import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('pos_2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('new_template.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
print (res)
threshold = 0.4
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.jpg',img_rgb)'''









































'''template = cv2.imread('new_template.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template,85,255)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

image = cv2.imread('pos_3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
 
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])
    
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break

    edged = cv2.Canny(resized, 85, 255)
    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    
    found = (maxVal, maxLoc, r)

#print("Template Matched")    
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)'''