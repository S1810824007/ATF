import cv2
import numpy as np
from util import draw_line

def canny():
    img = cv2.imread("data/porsche.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough(image, threshold, same_line_thresh):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
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

    matrix_d_a = []
    for (d, angle) in np.argwhere(H >= threshold):
        buff = []
        if d > max_d:
            d = d - range_d

        buff.append(d)
        buff.append(angle)
        matrix_d_a.append(buff)

    print(matrix_d_a)

    if len(matrix_d_a) > 1:
        index = 0

        same_line_d_a = []
        while index in range(0, len(matrix_d_a)):
            similar = False
            if index != len(matrix_d_a) - 1:
                if matrix_d_a[index + 1][0] in range(matrix_d_a[index][0] - same_line_thresh, matrix_d_a[index][0] + same_line_thresh):
                    if matrix_d_a[index + 1][1] in range(matrix_d_a[index][1] - same_line_thresh, matrix_d_a[index][1] + same_line_thresh):
                        similar = True

            buff = []
            buff.append(matrix_d_a[index][0])
            buff.append(matrix_d_a[index][1])
            same_line_d_a.append(buff)

            if not similar:
                print("\nSame lines: ", same_line_d_a)

                # now print the line with the highes vote
                max = 0
                current_max = 0

                for i in range(0, len(same_line_d_a)):
                    h = H[same_line_d_a[i][0], same_line_d_a[i][1]]
                    if h > current_max:
                        current_max = h
                        max = same_line_d_a[i]

                print("Highes Vote: ", H[max[0], max[1]], " ", max)
                draw_line(img, max[0], np.deg2rad(max[1]))

                same_line_d_a = []

            index += 1

    else:
        draw_line(img, d, np.deg2rad(angle))

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H

if __name__ == '__main__':
    hough("data/triangle.png", 0.4, 5)




