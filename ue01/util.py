import cv2
import numpy as np

# Import this module to your code using "import util" and call the function using "util.draw_line(...)".


def draw_line(image, d, theta, color=(0, 0, 255)):
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * d
    y0 = b * d

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(image, (x1, y1), (x2, y2), color, 1)


if __name__ == '__main__':
    # demo (is not executed when you import this module elsewhere)
    img = np.zeros((300, 400, 3), dtype=np.uint8)  # black image
    draw_line(img, d=200, theta=np.deg2rad(30))
    draw_line(img, d=-10, theta=np.deg2rad(170), color=(0, 255, 0))
    draw_line(img, d=150, theta=np.deg2rad(90), color=(255, 255, 255))
    cv2.imshow("Demo for Draw Polar Line", img)
    cv2.waitKey(0)
