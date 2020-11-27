import cv2
import numpy as np
from util import draw_line

def filterSameLine(matrix_d_a, H, same_line_thresh_d, same_line_thresh_a, max_angle):
    SAME_LINES = []
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

                # check whole matrix for identical lines
                while j < len(matrix_d_a):
                    if matrix_d_a[j] not in already_used:

                        if current[1] - same_line_thresh_d < 0:
                            # current = [45, 2]
                            # any line: [-38, 179]

                            buffer_d = current[0]
                            if matrix_d_a[j][0] < 0 or current[0] < 0:
                                buffer_d = buffer_d * (-1)

                            if matrix_d_a[j][0] in range(buffer_d - same_line_thresh_d, buffer_d + same_line_thresh_d):
                                # buffer_d = -45, if -38 in range (buffer_d +- threshold)
                                same_line = False
                                if matrix_d_a[j][1] in range(current[1], current[1] + same_line_thresh_a):
                                    # if 179 in range 2 : 2 + threshold
                                    # current line = any line
                                    # identical lines and not used already -> add them to same_line_d_a
                                    same_line = True
                                if matrix_d_a[j][1] in range(0, current[1]):
                                    # if 179 in range 0 : 2
                                    # 0 - threshold ->
                                    # [45, 2] -> 2 : 2+threshold || 0-2 || 179 - threshold + a : 179
                                    # [0, 1, 2], 179-5+2 -> 176 : 179
                                    same_line = True
                                if matrix_d_a[j][1] in range(max_angle - same_line_thresh_a + current[1], max_angle):
                                    # anyline = 179
                                    # range (176 : 179)
                                    same_line = True
                                if same_line:
                                    buff = matrix_d_a[j]
                                    already_used.append(buff)
                                    same_line_d_a.append(buff)
                        elif matrix_d_a[j][0] in range(current[0] - same_line_thresh_d,
                                                       current[0] + same_line_thresh_d):
                            if matrix_d_a[j][1] in range(current[1] - same_line_thresh_a,
                                                         current[1] + same_line_thresh_a):
                                # identical lines and not used already -> add them to same_line_d_a
                                buff = matrix_d_a[j]
                                already_used.append(buff)
                                same_line_d_a.append(buff)
                    j += 1

            # now identify the highest vote and print the line
            if same_line_d_a != []:
                # print("Same lines: \n", same_line_d_a)
                SAME_LINES.append(same_line_d_a[0])

            # # now identify the highest vote and print the line
            # max = 0
            # current_max = 0
            # if same_line_d_a != []:
            #     #print("\nSame lines: ", same_line_d_a)
            #     for i in range(0, len(same_line_d_a)):
            #         #draw_line(img, same_line_d_a[i][0], np.deg2rad(same_line_d_a[i][1]), (150,150,150))
            #         h = H[same_line_d_a[i][0], same_line_d_a[i][1]]
            #         if h > current_max:
            #             current_max = h
            #             max = same_line_d_a[i]
            #
            #     #print("Highes Vote: ", H[max[0], max[1]], " ", max)
            #     SAME_LINES.append(max)
            index += 1
    return SAME_LINES

def canny():
    img = cv2.imread("data/porsche.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough(image, threshold, same_line_thresh_d, same_line_thresh_a, street_lane_thresh, STEPS=False):
    img = cv2.resize(image, (640, 360))
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

    buff = np.argwhere(canny != 0)
    x = buff[:, 1]
    y = buff[:, 0]

    H = np.zeros([range_d // step_d, max_angle // step_angle], dtype=np.uint8)

    for angle in range(0, max_angle, step_angle):
        angle_rad = np.deg2rad(angle)
        d = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        d = d.astype(int)
        d[d < 0] = range_d + d[d < 0]
        test = np.bincount(d)
        H[d, angle] = H[d, angle] + test[d]

    matrix_d_a = []
    for (d, angle) in np.argwhere(H >= threshold):
        if d > max_d:
            d = d - range_d
        matrix_d_a.append([d,angle,H[int(d), int(angle)]])

    matrix_d_a = np.array(matrix_d_a)
    matrix_d_a = matrix_d_a[matrix_d_a[:, 2].argsort()]
    matrix_d_a = matrix_d_a[::-1]
    matrix_d_a = matrix_d_a[:,0:2].tolist()

    SAME_LINES = filterSameLine(matrix_d_a, H, same_line_thresh_d, same_line_thresh_a, max_angle)

    right_x = 0
    right_line = False
    left_x = 0
    left_line = False
    for max in SAME_LINES:
        if (right_line and left_line):
            break
        x = (max[0] - (len(img) - 1) * np.sin(np.deg2rad(max[1]))) / np.cos(np.deg2rad(max[1]))
        if STEPS:
            draw_line(img, max[0], np.deg2rad(max[1]), (200, 200, 200))
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if (x > 0 and x < len(img[0]) * 1.2):
            if (max[1] in range(120 - street_lane_thresh, 120 + street_lane_thresh) and not right_line):
                print("right: ", max)
                right_x = x - (len(img[0]) / 2)
                draw_line(img, max[0], np.deg2rad(max[1]), (0, 0, 255))
                right_line = True
            elif (max[1] in range(60 - street_lane_thresh, 60 + street_lane_thresh) and not left_line):
                print("left: ", max)
                left_x = (len(img[0]) / 2) - x
                draw_line(img, max[0], np.deg2rad(max[1]), (0, 255, 0))
                left_line = True

    img = cv2.putText(img, "R: " + str(round(right_x)) + "px", (len(img[0]) // 2 + 100, len(img) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, "L: " + str(round(left_x)) + "px", (len(img[0]) // 2 - 100, len(img) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if STEPS:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def filterSameLineUE02(matrix_d_a, same_line_thresh_d, same_line_thresh_a, max_angle):
    SAME_LINES = []
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

                # check whole matrix for identical lines
                while j < len(matrix_d_a):
                    if matrix_d_a[j] not in already_used:

                        if current[1] - same_line_thresh_d < 0:
                            # current = [45, 2]
                            # any line: [-38, 179]

                            buffer_d = current[0]
                            if matrix_d_a[j][0] < 0 or current[0] < 0:
                                buffer_d = buffer_d * (-1)

                            if matrix_d_a[j][0] in range(buffer_d - same_line_thresh_d, buffer_d + same_line_thresh_d):
                                # buffer_d = -45, if -38 in range (buffer_d +- threshold)
                                same_line = False
                                if matrix_d_a[j][1] in range(current[1], current[1] + same_line_thresh_a):
                                    # if 179 in range 2 : 2 + threshold
                                    # current line = any line
                                    # identical lines and not used already -> add them to same_line_d_a
                                    same_line = True
                                if matrix_d_a[j][1] in range(0, current[1]):
                                    # if 179 in range 0 : 2
                                    # 0 - threshold ->
                                    # [45, 2] -> 2 : 2+threshold || 0-2 || 179 - threshold + a : 179
                                    # [0, 1, 2], 179-5+2 -> 176 : 179
                                    same_line = True
                                if matrix_d_a[j][1] in range(max_angle - same_line_thresh_a + current[1], max_angle):
                                    # anyline = 179
                                    # range (176 : 179)
                                    same_line = True
                                if same_line:
                                    buff = matrix_d_a[j]
                                    already_used.append(buff)
                                    same_line_d_a.append(buff)
                        elif matrix_d_a[j][0] in range(current[0] - same_line_thresh_d, current[0] + same_line_thresh_d):
                            if matrix_d_a[j][1] in range(current[1] - same_line_thresh_a, current[1] + same_line_thresh_a):
                                # identical lines and not used already -> add them to same_line_d_a
                                buff = matrix_d_a[j]
                                already_used.append(buff)
                                same_line_d_a.append(buff)
                    j += 1

            # now identify the highest vote and print the line
            if same_line_d_a != []:
                SAME_LINES.append(same_line_d_a[0])
            index += 1
    return SAME_LINES

def houghUE2(img, threshold, same_line_thresh_d, same_line_thresh_a, street_lane_thresh, STEPS=False):
    img = cv2.resize(img, (640, 360))

    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(canny, 200, 300)

    mask = cv2.imread("data/mask.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (640, 360))
    canny[mask == 0] = 0

    d_res = 1
    theta_res = np.pi / 180
    lines = cv2.HoughLines(canny, d_res, theta_res, threshold)

    lines[:,0,1] = np.rad2deg(lines[:,0,1])

    matrix_d_a = []
    for line in lines:
        for d, theta in line:
            matrix_d_a.append([int(d), int(theta)])

    # now filter same lines
    SAME_LINES = filterSameLineUE02(matrix_d_a, same_line_thresh_d, same_line_thresh_a, 180)

    right_x = 0
    right_line = False
    left_x = 0
    left_line = False
    for max in SAME_LINES:
        if(right_line and left_line):
            break
        x = (max[0] - (len(img)-1) * np.sin(np.deg2rad(max[1]))) / np.cos(np.deg2rad(max[1]))
        if STEPS:
            draw_line(img, max[0], np.deg2rad(max[1]), (200, 200, 200))
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if(x > 0 and x < len(img[0]) * 1.2):
            if(max[1] in range(120-street_lane_thresh, 120 + street_lane_thresh) and not right_line):
                print("right: ", max)
                right_x = x - (len(img[0]) / 2)
                draw_line(img, max[0], np.deg2rad(max[1]), (0, 0, 255))
                right_line = True
            elif(max[1] in range(60-street_lane_thresh, 60 + street_lane_thresh) and not left_line):
                print("left: ", max)
                left_x = (len(img[0]) / 2) - x
                draw_line(img, max[0], np.deg2rad(max[1]), (0, 255, 0))
                left_line = True

    img = cv2.putText(img, "R: " + str(round(right_x)) + "px", (len(img[0]) // 2 + 100, len(img) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, "L: " + str(round(left_x)) + "px", (len(img[0]) // 2 - 100, len(img) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if STEPS:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def video(thresh_hough, thresh_same_d, thresh_same_a, thresh_a):
    cap = cv2.VideoCapture("data/highway1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = houghUE2(frame, thresh_hough, thresh_same_d, thresh_same_a, thresh_a)
        #img = hough(frame, thresh_hough, thresh_same_d, thresh_same_a, thresh_a)
        cv2.imshow("Result", img)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    thresh_hough = 40       #40 80
    thresh_same_d = 50      #50 20
    thresh_same_a = 15      #15 15
    thresh_a = 25           #25 20

    #img = cv2.imread("data/triangle.png")
    img = cv2.imread("data/highway1-1.png")

    hough(img, thresh_hough, thresh_same_d, thresh_same_a, thresh_a, True)
    #houghUE2(img, thresh_hough, thresh_same_d, thresh_same_a, thresh_a, True)

    #video(thresh_hough, thresh_same_d, thresh_same_a, thresh_a)