import cv2
import numpy as np

SAMPLE_SIZE = 50
MIN_CONTOUR_AREA = 1000

# Used in cv2.HoughLinesP()
MIN_LINE_LENGTH = 2
MAX_LINE_GAP = 20

lanes = []

""" segmentation_v2
Given a preprocessed binary image of a road, this function will segment the road
into polygons, with each polygon representing the area of one lane. It will then
return an image in which the lane areas are filled in with different colors.
The boundaries of each area are stored as an element in the "lanes" list.

This function can handle any number of distinct lanes, and the amount of lanes
does not have to be specified. It can also handle curved and straight roads
alike, although it is designed to work on mostly vertical roads. In the future 
I'd like this function to also handle more horizontal roads.

The process of segmentation is as follows:
- find the rough shapes in the image (contours)
- remove the small shapes and then sort them by their positions on the x-axis
- use Hough Transform to find the lines that represent each lane and assign
  them to the appropriate contour
- now that the lines are grouped together, for each group, find the best-fit
  polynomial from the lines' points to represent a lane boundary (the white and
  yellow lines around a lane).
- Finally, get points from these polynomials and fill in the regions in between
  them (thereby filling in the area of the lanes).

This function takes some inspiration from the this paper:
https://www.ingentaconnect.com/contentone/ist/ei/2016/00002016/00000014/art00011?crawler=true
"""
def segmentation_v2 (binary):
    global lanes

    kernel = np.ones((4,4), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)

    # Build the output image
    out = np.dstack(( np.zeros_like(dilation), 
            np.zeros_like(dilation), 
            np.zeros_like(dilation) )) * 255

    _,contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, 
                                cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    # For mostly vertical lanes, we want contours with similar x-coordinates
    # to be grouped together. For mostly horizontal lanes, we'd use 
    # y-coordinates instead.
    sorted_ctrs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0] )
    
    # Use Hough Transform to find rough lines that represent each lane.
    linesP = cv2.HoughLinesP(dilation, 1, np.pi / 180, 20, None, 
                MIN_LINE_LENGTH, MAX_LINE_GAP)
    
    # Match up Hough lines with nearby contours
    groups = assign_groups(linesP, sorted_ctrs)
    
    # From each group, sample 50 points to represent a lane border; the x and y
    # coordinates will be stored in border_x and border_y, respectively.
    border_x, border_y = sample_points(groups)
    
    assign_lanes(out, border_x, border_y)

    # Draw the contours themselves, for testing purposes    
    cv2.drawContours(out, sorted_ctrs, -1, (0,255,0))

    return out

""" assign_groups
Given a list of Hough Lines and list of contours, this function 'assigns' the
lines to a certain contour. We do this step because a single lane may be
represented by multiple lines (for example, a curved lane). 
"""
def assign_groups (linesP, sorted_ctrs):

    groups = [None]*len(sorted_ctrs)

    if linesP is not None:
        for i in range(0, len(linesP)):
            x1,y1,x2,y2 = linesP[i][0]

            line = [x1,y1,x2,y2] if y1 > y2 else [x2,y2,x1,y1]
            mid = ( (x1+x2)//2, (y1+y2)//2 )
            
            for j in range(len(sorted_ctrs)):
                # If the midpoint of this line is inside a contour 0, then this
                # line is said to be part of group 0. This line will be check 
                # against all of the contours and assigned to a group 
                # accordingly.
                if cv2.pointPolygonTest(sorted_ctrs[j], mid, True) >= 0:
                    if groups[j] is not None:
                        groups[j] = np.append(groups[j], [line], axis=0)
                    else:
                        groups[j] = np.array([line])
                    break

    return groups

""" sample_points
Now that we have the lines grouped into their lanes, we use the points of these
lines to generate a best-fit polynomial of each lane. After generating the
polynomial, we sample 50 of its points and push them to border_x, and border_y,
respectively.
"""
def sample_points(groups):
    border_x = []
    border_y = []

    for lines in groups:

        if lines is None:
            continue

        # Separate the x_coordinates and y coordinates of the lines
        x_pos = np.array([[x1,x2] for x1,_,x2,_ in lines]).ravel()
        y_pos = np.array([[y1,y2] for _,y1,_,y2 in lines]).ravel()
        
        # Use these x and y coords to find a best-fit quadratic, which will 
        # represent the edge of a lane.
        fit = np.polyfit(x_pos,y_pos,2)
        
        # Then, sample 50 points from this polynomial and append them to 
        # border_x and border_y
        draw_x = np.linspace(np.min(x_pos), np.max(x_pos), num=SAMPLE_SIZE)
        draw_y = np.polyval(fit, draw_x)

        border_x = np.append(border_x, draw_x)
        border_y = np.append(border_y, draw_y)

    return (border_x, border_y)


""" assign_lanes
Now that we have polynomial coordinates for each group, we can organize them
into points, and those points into lanes. We then fill in each distinct lane
with a random color. The final areas of each lane are stored in the global 
"lanes" list.
"""
def assign_lanes(out, border_x, border_y):
    global lanes

    # Pair up the x-coordinates and y-coordinates accordingly
    pts = np.int32( np.column_stack((border_x, border_y)) )

    # Because every polynomial is made up of 50 points, that means that every
    # 50 points in the array there is a new polynomial, i.e. a new border.
    # We join two adjacent borders together to make a lane and then
    # append it to lanes[].
    for i in range(0,len(pts)-SAMPLE_SIZE,SAMPLE_SIZE):
        lane = np.concatenate(( pts[i:i+SAMPLE_SIZE], 
                np.flip(pts[i+SAMPLE_SIZE:i+SAMPLE_SIZE*2], axis=0) ))
        lanes.append(lane)

    # Fill in the the lanes' areas in a random color, 
    # for demonstration and testing purposes.
    for lane in lanes:
        color = (np.random.randint(255), 
                    np.random.randint(255), np.random.randint(255))
        cv2.fillPoly(out, np.array([lane]), color)