import cv2
import numpy as np

# Lane line parameters
avg_left_line_slope = 1
avg_left_line_intercept = 1
avg_right_line_slope = 1
avg_right_line_intercept = 1

def lane_line_detection(frame):
    # Set global variables
    global avg_left_line_slope, avg_left_line_intercept, avg_right_line_slope, avg_right_line_intercept
    # Read image
    #image = mpimg.imread('images/test2.jpg')
    image = frame
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
    canny_img = cv2.Canny(blur_gray,low_thresh,high_thresh) # outputs a binary image

    # Mask
    margin = 100
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


    left_line_slopes = []
    left_line_intercepts = []
    right_line_slopes = []
    right_line_intercepts = []


    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2-y1) / (x2-x1)
        intercept = y1 - (slope * x1)
        if slope < 0.:
            left_line_slopes.append(slope)
            left_line_intercepts.append(intercept)
        elif slope > 0.:
            right_line_slopes.append(slope)
            right_line_intercepts.append(intercept)


    if len(left_line_slopes)>0 and len(left_line_intercepts)>0:
        avg_left_line_slope = np.mean(np.array(left_line_slopes))
        avg_left_line_intercept = np.mean(np.array(left_line_intercepts))

    if len(right_line_slopes)>0 and len(right_line_intercepts)>0:
        avg_right_line_slope = np.mean(np.array(right_line_slopes))
        avg_right_line_intercept = np.mean(np.array(right_line_intercepts))



    y_min = int(y_size/2+margin)
    y_max = y_size

    # Compute left lane line
    left_x_min = int((y_min-avg_left_line_intercept) / avg_left_line_slope)
    left_x_max = int((y_max-avg_left_line_intercept) / avg_left_line_slope)
    

    # Compute right lane line
    right_x_min = int((y_min-avg_right_line_intercept) / avg_right_line_slope)
    right_x_max = int((y_max-avg_right_line_intercept) / avg_right_line_slope)


    # Create lane image
    lane_lines_img = np.zeros_like(img_copy)
    cv2.line(lane_lines_img,(left_x_min,y_min),(left_x_max,y_max),[0,0,255],12)
    cv2.line(lane_lines_img,(right_x_min,y_min),(right_x_max,y_max),[0,0,255],12)


    # Blend images
    out_img = cv2.addWeighted(img_copy, 1, lane_lines_img, 0.8, 0)


    return out_img



# Video reader
cap = cv2.VideoCapture('input/solidWhiteRight.mp4')

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi',fourcc,20.0  ,(int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        frame = lane_line_detection(frame)
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(25) and 0xFF==ord('q'):
            break

    else:
        break


# Release all
cap.release()
out.release()
cv2.destroyAllWindows()


        
        



