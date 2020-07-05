import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Read image
image = mpimg.imread('images/test2.jpg')
y_size = image.shape[0]
x_size = image.shape[1]
img_copy = np.copy(image)

# Grayscale
gray = cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)

# Gaussian blur
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

# Canny
low_thresh = 50
high_thresh = 150
canny_img = cv2.Canny(blur_gray,low_thresh,high_thresh) #outputs a binary image

# Mask
margin = 50
vertices = np.array([[0,y_size],[x_size/2-margin,y_size/2+margin],[x_size/2+margin,y_size/2+margin],[x_size,y_size]],np.int32)
mask = np.zeros_like(canny_img)
cv2.fillPoly(mask,[vertices],255)
masked_canny_img = np.bitwise_and(canny_img,mask)

# Hough lines
rho = 2
theta = np.pi/180
thresh = 15
min_line_lenth = 40
max_gap_bet_lines = 20
lines = cv2.HoughLinesP(masked_canny_img,rho,theta,thresh,np.array([]),min_line_lenth,max_gap_bet_lines)

# Lane line detection
left_line_xs = []
left_line_ys = []
right_line_xs = []
right_line_ys  = []

for line in lines:
    x1,y1,x2,y2 = line[0]
    slope = (y2-y1) / (x2-x1)
    if slope < 0.:
        left_line_xs.extend((x1,x2))
        left_line_ys.extend((y1,y2))
    elif slope > 0.:
        right_line_xs.extend((x1,x2))
        right_line_ys.extend((y1,y2))


left_line_params = np.polyfit(left_line_xs,left_line_ys,1)
right_line_params = np.polyfit(right_line_xs,right_line_ys,1)



y_min = 400
y_max = 540

# Draw left lane line
left_x_min = int((y_min-left_line_params[1]) / left_line_params[0])
left_x_max = int((y_max-left_line_params[1]) / left_line_params[0])
cv2.line(img_copy,(left_x_min,y_min),(left_x_max,y_max),[255,0,0],10)

# Draw left lane line
right_x_min = int((y_min-right_line_params[1]) / right_line_params[0])
right_x_max = int((y_max-right_line_params[1]) / right_line_params[0])
cv2.line(img_copy,(right_x_min,y_min),(right_x_max,y_max),[255,0,0],10)


# Plot image
plt.imshow(img_copy)
plt.show()
        
        



