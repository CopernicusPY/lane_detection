import cv2
import json
import numpy
import matplotlib.pylab as plt

with open('conf.json') as json_file:
    data = json.load(json_file)
THETA = 1 * numpy.pi / 180
# Initialize the image and create a duplicate to apply the filters on
img = cv2.imread(data['source'])
img_copy = img
cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

_, img_copy = cv2.threshold(img_copy, data['bin.low_threshold'], data['bin.high_threshold'], cv2.THRESH_BINARY)

img_copy =cv2.Canny(numpy.array(img_copy), data['canny.low_threshold'], data['canny.high_threshold'])
lines = cv2.HoughLinesP(img_copy, data['hough.rho'], THETA, data['hough.threshold'],numpy.array([]), data['hough.min_line_length'], data['hough.max_line_gap'])
# # Draw the detected lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    #Get the line.color from the config file and turn it in a tuple for the cv2.line function
    cv2.line(img, (x1, y1), (x2, y2),tuple(data['line.color']), data['line.thickness'])

plt.imshow(img)
plt.show()
