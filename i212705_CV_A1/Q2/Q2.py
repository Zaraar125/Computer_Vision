import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

chessboard_path='Question_2_images/chess_1.png'
chess_board=cv2.imread(chessboard_path,1)

# Convert to grayscale
gray = cv2.cvtColor(chess_board, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny Edge Detector
edges = cv2.Canny(gray, 50, 150)

# Detect lines using the probabilistic Hough Line Transform
linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=380, minLineLength=50, maxLineGap=10)

# Separate lines into vertical and horizontal based on their slope
vertical_lines = []
horizontal_lines = []

for line in linesP:
    x1, y1, x2, y2 = line[0]
    if abs(y1 - y2) < 10:  # Threshold to consider as horizontal
        horizontal_lines.append(((x1, y1, x2, y2)))
    
    # Check if the line is vertical (slope is undefined)
    elif abs(x1 - x2) < 10:  # Threshold to consider as vertical
        vertical_lines.append(((x1, y1, x2, y2)))

# finding unique lines 
horizontal_lines=horizontal_lines[:5]+horizontal_lines[6:]
# remove_index=[]
# cleaned=[]
# for i in range(len(horizontal_lines)-1):
#     x1,y1,x2,y2= horizontal_lines[i]
#     for j in range(i+1,len(horizontal_lines)):
#         x21,y21,x22,y22=horizontal_lines[j]
#         if abs(y22-y2)<=3 or abs(y21-y1)<=3:
#             remove_index.append(j)
#             break
#     if i not in remove_index:
#         cleaned.append((x1,y1,x2,y2))
# horizontal_lines=cleaned

# Draw lines on the image in the horizantal direction
for line in horizontal_lines:
    x1, y1, x2, y2 = line
    cv2.line(chess_board, (x1, y1), (x2, y2), (255, 0, 0), 2)
for line in vertical_lines:
    x1, y1, x2, y2 = line
    cv2.line(chess_board, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Function to find the intersection of two lines (vertical and horizontal)
def find_intersection(v_line, h_line):
    x1, y1, x2, y2 = v_line
    x3, y3, x4, y4 = h_line

    if x1 == x2:  # Vertical line
        if y3 == y4:  # Horizontal line
            return (x1, y3)
    return None

# Find intersections between vertical and horizontal lines
intersections = []
for v_line in vertical_lines:
    for h_line in horizontal_lines:
        intersection = find_intersection(v_line, h_line)
        if intersection is not None:
            intersections.append(intersection)

# Estimate the number of boxes based on intersections
num_horizontal_intersections = len(set([p[0] for p in intersections]))
num_vertical_intersections = len(set([p[1] for p in intersections])) 

# The number of boxes is one less than the number of intersections in each direction
num_boxes = (num_horizontal_intersections - 1) * (num_vertical_intersections - 1)

# Display the result
print(f"Number of boxes: {num_boxes}")

# join canny filter and transformed image
array_3d = np.expand_dims(edges, axis=-1)
zeros = np.zeros_like(array_3d)
result = np.concatenate((array_3d, zeros,zeros), axis=-1)
joined=np.concatenate([chess_board,result],1)

cv2.imshow('joined canny filter and detected corners',joined)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'Number of boxes detected: {num_boxes}')

