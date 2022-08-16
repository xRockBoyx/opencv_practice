from xml.dom import HierarchyRequestErr
import cv2 as cv

origin_img = cv.imread("/Users/huangweikai/Desktop/image.png", 
                       cv.IMREAD_UNCHANGED)

# ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
gray_img = cv.cvtColor(origin_img, 
                       cv.COLOR_BGR2GRAY)

ret, th1 = cv.threshold(gray_img, 
                        127, 
                        255, 
                        cv.THRESH_BINARY)

# print(ret)
Contours, Hierarchy = cv.findContours(th1, 
                                      cv.RETR_LIST, 
                                      cv.CHAIN_APPROX_SIMPLE)

for c in Contours:
    (x, y, w, h) = cv.boundingRect(c)
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    cv.putText(origin_img, 
               "Coordinate : " + str(center_x) + " " + str(center_y),
               (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX,
               1.2,
               (0, 0, 0),
               2)
# print(len(Contours))
# print('--------')
# print(Hierarchy)
cv.imshow("window", origin_img)
cv.waitKey(0)