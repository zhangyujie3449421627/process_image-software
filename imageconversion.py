from PyQt5.QtGui import QImage
import numpy as np


def ndarry2QImage(img):
    # https: // blog.csdn.net / weixin_43935474 / article / details / 113117853
    # QImage构造函数加了第4个参数int bytesPerLine，即每行字节数，否则部分图像转换异常
    if len(img.shape) == 2 and img.dtype == 'uint8':  #
        qImg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1],QImage.Format_Grayscale8)
        return qImg

    if img.dtype == 'uint8' and len(img.shape) == 3 and img.shape[2] == 4:
        qImg = QImage(img.data, img.shape[1], img.shape[0],img.shape[1]*4, QImage.Format_ARGB32)
        return qImg

    if img.dtype == 'uint8' and len(img.shape) == 3 and img.shape[2] == 3:
        qImg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        return qImg


# 参考资料：https://blog.csdn.net/yx1302317313/article/details/104527401
def QImage2ndarry(image):
    size = image.size()
    print(image.size)
    s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
    arr = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
    return arr

# img = cv2.imread('lena.jpg')
# img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print("test1:type(img)：",type(img))
# tempQImage = ndarry2QImage(img)
# print("test2:type(tempQImage):",type(tempQImage))
# tempndarry = QImage2ndarry(tempQImage)
# print("test3:type(tempQImage):",type(tempndarry))
# cv2.imshow("image",tempndarry)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
