import cv2
import numpy as np
from util import draw_line

def canny():
    img = cv2.imread("data/porsche.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough(image, threshold, same_line_thresh, STEPS=False):
    img = cv2.imread(image)
    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(canny, 100, 200)

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
        if d > max_d:
            d = d - range_d

        matrix_d_a.append([d,angle])

    print("\nAll lines: ", matrix_d_a)

    if len(matrix_d_a) > 1:
        index = 0

        already_used = []
        while index in range(0, len(matrix_d_a)):
            same_line_d_a = []
            current = matrix_d_a[index]
            if current not in already_used:
                same_line_d_a.append(current)
                already_used.append(current)
                j = index + 1

                while j < len(matrix_d_a):
                    if matrix_d_a[j] not in already_used:
                        if matrix_d_a[j][0] in range(current[0] - same_line_thresh, current[0] + same_line_thresh):
                            if matrix_d_a[j][1] in range(current[1] - same_line_thresh, current[1] + same_line_thresh):
                                # identical lines and not used already -> add them to same_line_d_a
                                buff = matrix_d_a[j]
                                already_used.append(buff)
                                same_line_d_a.append(buff)
                    j += 1

            # now print the line with the highes vote
            max = 0
            current_max = 0
            if same_line_d_a != []:
                print("\nSame lines: ", same_line_d_a)
                for i in range(0, len(same_line_d_a)):
                    #draw_line(img, same_line_d_a[i][0], np.deg2rad(same_line_d_a[i][1]), (255,0,0))
                    h = H[same_line_d_a[i][0], same_line_d_a[i][1]]
                    if h > current_max:
                        current_max = h
                        max = same_line_d_a[i]

                print("Highes Vote: ", H[max[0], max[1]], " ", max)
                draw_line(img, max[0], np.deg2rad(max[1]))

                if STEPS:
                    cv2.imshow(image, img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            index += 1

    else:
        draw_line(img, d, np.deg2rad(angle))

    cv2.imshow(image, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H

if __name__ == '__main__':
    hough("data/triangle.png", 0.3, 10, True)
    #hough("data/dice.png", 0.5, 25, True)




