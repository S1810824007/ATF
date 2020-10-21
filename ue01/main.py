import cv2
import numpy as np
import util as u

def canny():
    img = cv2.imread("data/porsche.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough():
    img = cv2.imread("data/triangle.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)

    step_d = 1
    step_angle = 1
    max_angle = 180

    x = len(img[0])
    y = len(img)

    min_d = -x
    max_d = np.sqrt(x * x + y * y)
    range_d = int(np.round(max_d - min_d))

    H = np.zeros([range_d // step_d, max_angle // step_angle], dtype=np.uint8)

    for (y, x) in np.argwhere(canny != 0):
        for angle in range(0, max_angle, step_angle):
            angle_rad = np.deg2rad(angle)
            d = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            H[int(d / step_d), angle // step_angle] += 1;

    H = H / H.max()

    threshold = 0.5
    for (d, angle) in np.argwhere(H >= threshold):
        if d > max_d:
            d = d - range_d
        print(d, " ", angle)
        u.draw_line(img, d, np.deg2rad(angle))

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough()


