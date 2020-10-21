import cv2

if __name__ == '__main__':
    img = cv2.imread("data/porsche.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)

    cv2.imshow("canny", canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()