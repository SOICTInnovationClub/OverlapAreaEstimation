import cv2
import numpy as np
from shapely import Polygon
import copy

# Đọc ảnh binary
image = cv2.imread('final.jpg', cv2.IMREAD_GRAYSCALE)
# lấy kích thước ảnh
height, width = image.shape[:2]

# cắt ảnh thành 4 phần bằng cách chia đôi chiều rộng và chiều cao
x_center = width // 2
y_center = height // 2

cam1 = image[0:y_center, 0:x_center]
cam2 = image[0:y_center, x_center:width]
cam3 = image[y_center:height, 0:x_center]
cam4 = image[y_center:height, x_center:width]
# Tìm contours
contours1, hierarchy1 = cv2.findContours(cam1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy2 = cv2.findContours(cam2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy3 = cv2.findContours(cam3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours4, hierarchy4 = cv2.findContours(cam4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow("cam", cam3)
# cv2.waitKey(-1)
# cv2.drawContours(cam3, contours3, -1, (0, 255, 0), 30)
# cv2.imshow("contour", cam3)
# cv2.waitKey(-1)
# breakpoint()
# Tạo mask để fill vùng đen
mask1 = np.zeros_like(cam1)
mask2 = np.zeros_like(cam2)
mask3 = np.zeros_like(cam3)
mask4 = np.zeros_like(cam4)

# Fill các Convex Hull

def find_maxpolygon(contours):
    max_pol, max_area = None, 0

    for contour in contours:
        hull1 = cv2.convexHull(contour)
        # cv2.fillConvexPoly(mask1, hull1, color=(255, 255, 255))

        hull1 = hull1.reshape(-1, 2)
        try:
            pol_area = Polygon(hull1).buffer(0).area
        except:
            pol_area = 0

        if max_area < pol_area:
            max_area = pol_area
            max_pol = hull1
            # breakpoint()

        # breakpoint()
    return max_pol


pol1 = find_maxpolygon(contours1)
pol2 = find_maxpolygon(contours2)
pol3 = find_maxpolygon(contours3)
pol4 = find_maxpolygon(contours4)
f = open(f"Cam_{1}.txt", "w")
s = "("
for i in range(len(pol1)):
    s += str(pol1[i][0]) + ',' + str(pol1[i][1]) + ','
s = s[:-1] + ")"
print(s)
