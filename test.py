import cv2
import numpy as np


def show_image(desc, image):
    cv2.imshow(desc, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reg_area_color(image):
    """找到原图像最多的颜色，当该颜色为红色或蓝色时返回该颜色的名称"""
    kernel = np.ones((35, 35), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 以上为图像处理
    Open = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)
    # 对Open图像的H通道进行直方图统计
    hist = cv2.calcHist([Open], [0], None, [180], [0, 180])
    # 找到直方图hist中列方向最大的点hist_max
    hist_max = np.where(hist == np.max(hist))

    # hist_max[0]为hist_max的行方向的值，即H的值，H在0~10为红色
    if 0 < hist_max[0] < 10:
        res_color = 'red'
    elif 100 < hist_max[0] < 124:  # H在100~124为蓝色
        res_color = 'blue'
    else:
        # H不在前两者之间跳出函数
        res_color = 'unknow'
    return res_color


img = cv2.imread('images2021/chepai.jpg')

# 调整图片大小
img = cv2.resize(img, (1024, 768))

# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image('gray', gray)

# 双边滤波
blf = cv2.bilateralFilter(gray, 13, 15, 15)
show_image('bilateralFilter', blf)

# 边缘检测
edged = cv2.Canny(blf, 30, 200)
show_image('canny', edged)

# 寻找轮廓（图像矩阵，输出模式，近似方法）
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 根据区域大小排序取前十
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
# 遍历轮廓，找到车牌轮廓
for c in contours:
    if cv2.contourArea(c) > 1024 * 768 * 0.05:
        continue

    # 计算轮廓周长（轮廓，是否闭合）
    peri = cv2.arcLength(c, True)
    # 折线化（轮廓，阈值（越小越接近曲线），是否闭合）返回折线顶点坐标
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # 获取四个顶点（即四边形, 左下/右下/右上/左上
    if len(approx) == 4:
        # [参数]左上角纵坐标:左下角纵坐标,左上角横坐标:右上角横坐标
        crop_image = img[approx[3][0][1]:approx[0][0][1], approx[3][0][0]:approx[2][0][0]]
        show_image('crop', crop_image)
        if 'blue' == reg_area_color(crop_image):
            screenCnt = approx
            break
# 如果找到了四边形
if screenCnt is not None:
    # 根据四个顶点坐标对img画线(图像矩阵，轮廓坐标集，轮廓索引，颜色，线条粗细)
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    show_image('contour', img)

"""遮罩"""
# 创建一个灰度图一样大小的图像矩阵
mask = np.zeros(gray.shape, np.uint8)
# 将创建的图像矩阵的车牌区域画成白色
cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
# 图像位运算进行遮罩
mask_image = cv2.bitwise_and(img, img, mask=mask)
show_image('mask_image', mask_image)

"""图像剪裁"""
# 获取车牌区域的所有坐标点
(x, y) = np.where(mask == 255)
# 获取底部顶点坐标
(topx, topy) = (np.min(x), np.min(y))
# 获取底部坐标
(bottomx, bottomy,) = (np.max(x), np.max(y))
# 剪裁
cropped = gray[topx:bottomx, topy:bottomy]
show_image('cropped', cropped)
