from menu import Ui_MainWindow
from imageconversion import ndarry2QImage
from PyQt5 import QtGui
import sys
import cv2
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QMessageBox,QGraphicsView,QPushButton,QButtonGroup
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
from PyQt5.QtCore import Qt
# 定义数字图像处理窗口类
class DIPWindow(QMainWindow, Ui_MainWindow):  # 定义类# 继承QMainWindow和Ui_MainWindow
    # 定义构造方法
    def __init__(self, parent=None):
        super(DIPWindow, self).__init__(parent)
        self.setupUi(self)# 设置UI界面
        # 获取当前窗体尺寸
        self.desktop = QApplication.desktop()# 获取桌面对象
        self.screenRect = self.desktop.screenGeometry()#获取屏幕的几何信息
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.7)#计算窗口高度
        self.width = int(self.screenwidth * 0.7)# 计算窗口宽度
        self.initial_size_height= self.height#存储初始的高
        self.initial_size_width = self.width#存储宽
        self.initial_pos = self.pos()  # 保存初始位置
        self.resize(self.width, self.height)#重新布局
        self.image ="" # 初始化图像变量
        self.image2=""# 初始化第二个图像变量
        # 添加动作
        # 连接action_start的点击信号到槽函数
        # 为menu_2添加一个动作并连接到槽函数
        self.action_show.triggered.connect(self.show_page_1)# 显示页面1
        self.action_show2.triggered.connect(self.show_page_2)# 显示页面2
        self.action_start.triggered.connect(self.openimage)# 打开图像
        self.action_save.triggered.connect(self.saveimage)# 保存图像
        self.action_quan.triggered.connect(self.show_quan)# 全屏显示
        self.action_quan1.triggered.connect(self.increase_size_150)# 增加大小到150%
        self.action_exit_fullscreen.triggered.connect(self.exit_fullscreen)# 退出全屏
        self.action_clear.triggered.connect(self.clear_all)# 清空画布
        self.action_hf.triggered.connect(self.restore_ui)#恢复原来窗口
        self.action_clear_graphics.triggered.connect(self.clear_graphics)# 清空缩略图
        # 连接comboBox的信号到槽函数
        self.comboBox.currentIndexChanged.connect(self.flip_image)# 翻转图像
        self.comboBox_2.currentIndexChanged.connect(self.lvjing_image) # 滤镜图像
        self.pushButton_qd.clicked.connect(self.pushbutton_xuanzhuan)# 旋转按钮

        # 初始化QGraphicsScene
        self.scene = QGraphicsScene(self) # 创建场景
        self.graphicsView.setScene(self.scene)# 设置视图的场景

        # 连接按钮的点击信号到槽函数
        self.pushButton_openimage1.clicked.connect(self.open_image)
        # 创建按钮组
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radioButton_1, 1)  # 参数1是按钮的ID
        self.button_group.addButton(self.radioButton_2, 2)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_3, 3)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_4, 4)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_5, 5)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_6, 6)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_7, 7)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_8, 8)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_9, 9)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_10, 10)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_11, 11)  # 参数2是按钮的ID
        self.button_group.addButton(self.radioButton_12, 12)  # 参数2是按钮的ID
        # 连接按钮的点击信号到槽函数
        self.pushButton_process.clicked.connect(self.pushbutton_process)#一堆radiobutton的处理按钮
        self.pushButton_qd1.clicked.connect(self.pushbutton_caijian)#裁剪按钮
        self.pushButton_qd1_2.clicked.connect(self.resize_image)#缩放图片按钮
        self.pushButton_rh.clicked.connect(self.image_merge)#图片融合
        self.pushButton_4.clicked.connect(self.addtext)#文字添加
        self.pushButton_3.clicked.connect(self.image_diejia)#图片叠加
        self.pushButton_2.clicked.connect(self.chepai)#车牌识别
        self.pushButton_5.clicked.connect(self.image_xiufu)#图像修复
        self.pushButton_6.clicked.connect(self.image_koutu)#图像抠图
        self.pushButton_7.clicked.connect(self.image_increase_color)#色彩增强
        self.pushButton_11.clicked.connect(self.image_liantongyu_detect)#连通域检测
        self.pushButton_9.clicked.connect(self.image_lunkuo)#图片轮廓检测
        self.pushButton_8.clicked.connect(self.shift_detect)#shift检测
        self.pushButton_14.clicked.connect(self.jiaoyanzaosheng)#椒盐噪声
        self.pushButton_15.clicked.connect(self.add_gaussian_noise)#高斯噪声添加
        self.pushButton_12.clicked.connect(self.Jinzita)#金字塔
        self.pushButton_13.clicked.connect(self.Toushi)#透视

    def restore_ui(self):
        # 重置窗口大小和位置
        self.resize(self.initial_size_width,self.initial_size_height)
        self.move(self.initial_pos)
        # 确保窗口居中
        self.move(self.screen().availableGeometry().center() - self.rect().center())
    def chepai(self):
        # 提取车牌（形态学）
        def Morph_Distinguish(img):
            # 1、转灰度图
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

            # 2、顶帽运算突出图像中的小亮区域，这些区域比周围区域亮，但并不比图像中所有区域都亮。这种运算通过从原始图像中减去开运算
            # gray = cv.equalizeHist(gray)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 17))#这行代码创建了一个结构元素（kernel），这是一个17x17像素的矩形。结构元素用于定义形态学操作的邻域大小和形状。在这里，它定义了顶帽运算中考虑的邻域。
            tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)#第二个参数是表面执行的是顶帽运算

            # 3、Sobel算子提取y方向边缘（揉成一坨）
            y = cv.Sobel(tophat, cv.CV_16S, 1, 0)
            absY = cv.convertScaleAbs(y)

            # 4、自适应二值化（阈值这里设置成75，效果还行，absY中的像素值大于75时，它们才会被转换为255（白色）255：最大值（max value），当像素值超过阈值时，这些像素值将被设置为这个值。）
            ret, binary = cv.threshold(absY, 75, 255, cv.THRESH_BINARY)

            # 5、开运算分割（纵向去噪，分隔）
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
            Open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

            # 6、闭运算合并，把图像闭合、揉团，使图像区域化，便于找到车牌区域，进而得到轮廓
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (41, 15))
            close = cv.morphologyEx(Open, cv.MORPH_CLOSE, kernel)

            # 7、膨胀/腐蚀（去噪得到车牌区域）
            # 中远距离车牌识别
            kernel_x = cv.getStructuringElement(cv.MORPH_RECT, (25, 7))
            kernel_y = cv.getStructuringElement(cv.MORPH_RECT, (1, 11))
            # 近距离车牌识别
            # kernel_x = cv.getStructuringElement(cv.MORPH_RECT, (79, 15))
            # kernel_y = cv.getStructuringElement(cv.MORPH_RECT, (1, 31))
            # 7-1、腐蚀、膨胀（去噪）
            erode_y = cv.morphologyEx(close, cv.MORPH_ERODE, kernel_y)
            dilate_y = cv.morphologyEx(erode_y, cv.MORPH_DILATE, kernel_y)

            # 7-1、膨胀、腐蚀（连接）（二次缝合）
            dilate_x = cv.morphologyEx(dilate_y, cv.MORPH_DILATE, kernel_x)

            erode_x = cv.morphologyEx(dilate_x, cv.MORPH_ERODE, kernel_x)

            # 8、腐蚀、膨胀：去噪
            kernel_e = cv.getStructuringElement(cv.MORPH_RECT, (25, 9))
            erode = cv.morphologyEx(erode_x, cv.MORPH_ERODE, kernel_e)

            kernel_d = cv.getStructuringElement(cv.MORPH_RECT, (25, 11))
            dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, kernel_d)

            # 9、获取外轮廓
            img_copy = img.copy()
            # 9-1、得到轮廓，使用OpenCV的cv.findContours函数来检测图像dilate中的轮廓
            # 函数返回两个值：contours和hierarchy。contours是一个列表，包含检测到的轮廓，每个轮廓都是一个点集。hierarchy是一个同样长度的列表，包含每个轮廓的层级信息，可以用来确定轮廓之间的父子关系
            contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # 9-2、画出轮廓并显示，使用OpenCV的cv.drawContours函数在图像img_copy上绘制轮廓，img_copy：要绘制轮廓的图像。
            # contours：要绘制的轮廓列表。
            # -1：指定绘制所有轮廓。
            cv.drawContours(img_copy, contours, -1, (255, 0, 255), 2)

            # 10、遍历所有轮廓，找到车牌轮廓
            i = 0
            for contour in contours:
                # 10-1、得到矩形区域：左顶点坐标、宽和高
                rect = cv.boundingRect(contour)
                # 10-2、判断宽高比例是否符合车牌标准，截取符合图片
                if rect[2] > rect[3] * 3 and rect[2] < rect[3] * 7:
                    # 截取车牌并显示
                    print(rect)
                    img = img[(rect[1] - 5):(rect[1] + rect[3] + 5), (rect[0] - 5):(rect[0] + rect[2] + 5)]  # 高，宽
                    try:
                        cv.imshow('license plate%d-%d' % (count, i), img)

                        i += 1
                    except:
                        pass
            return img_copy
        # if __name__ == '__main__':

        count = 0
        chepai = self.image
        # 3、定位车牌
        img =Morph_Distinguish(chepai)  # 形态学提取车牌调用函数
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(img)))
        cv.waitKey(0)
    def Toushi(self):#透视
        rows, cols = self.image.shape[:2]#获取输入图像self.image的行数（高度）和列数（宽度）。self.image.shape返回一个包含图像维度的元组，[:2]切片操作取出前两个元素，即高度和宽度。
        pts1 = np.float32([[150, 50], [400, 50], [60, 450], [310, 450]])
        pts2 = np.float32([[50, 50], [rows - 50, 50], [50, cols - 50], [rows - 50, cols - 50]])
        M = cv2.getPerspectiveTransform(pts1, pts2)#cv2.getPerspectiveTransform函数计算从pts1到pts2的透视变换矩阵M。
        self.dst = cv2.warpPerspective(self.image, M, (cols, rows))#使用OpenCV的cv2.warpPerspective函数应用透视变换。它将变换矩阵M应用于输入图像self.image，输出变换后的图像self.dst。(cols, rows)指定了输出图像的尺寸，与输入图像相同
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.dst)))
    def Jinzita(self):
        # 图像金字塔
        def gauss_image(image):
            level = 3
            img = image.copy()
            gauss_images = []
            gauss_images.append(G0)
            cv2.imshow("Gauss_0", G0)
            for i in range(level):
                dst = cv2.pyrDown(img)#cv2.pyrDown函数用于将图像尺寸减半
                gauss_images.append(dst)
                windName = 'Guass_{}'.format(i + 1)
                cv2.namedWindow(windName, 1)
                cv2.imshow(windName, dst)
                img = dst.copy()
            return gauss_images
        def laplian_image(image):
            gauss_images = gauss_image(image)
            level = len(gauss_images)
            for i in range(level - 1, 0, -1):
                expand = cv2.pyrUp(gauss_images[i], dstsize=gauss_images[i - 1].shape[:2])
                lpls = cv2.subtract(gauss_images[i - 1], expand)
                cv2.imshow('Laplacian_{}'.format(level - i), lpls)
            expand = cv2.pyrUp(cv2.pyrDown(gauss_images[3]), dstsize=gauss_images[3].shape[:2])
            lpls = cv2.subtract(gauss_images[3], expand)
            cv2.imshow('Laplacian_{}'.format(0), lpls)
        if __name__ == '__main__':
            # G0 = self.image#修改处,发现可以通过下面这两步，把通道顺序改一下就行了，格式就对了
            b, g, r = cv2.split(self.image)
            G0 = cv2.merge([r, g, b])
            if G0 is None:
                print('Failed to read image.')
                sys.exit()
            laplian_image(G0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # 彩色图像添加高斯噪声
    def add_gaussian_noise(self):
        # 获取图像的尺寸和通道数
        rows, cols, ch = self.image.shape
        # 生成高斯噪声
        gauss = np.random.normal(0, 25, (rows, cols, ch))
        # 将高斯噪声添加到图像上
        noisy_image = self.image + gauss
        # 确保像素值在0-255范围内
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)#限制范围
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(noisy_image)))
    # 彩色图像添加椒盐噪声
    def jiaoyanzaosheng(self):
        def add_noisy(image, n=10000):
            result = image.copy()
            w, h = image.shape[:2]
            for i in range(n):
                # 宽和高的范围内生成一个随机值，模拟表x,y坐标
                x = np.random.randint(1, w)
                y = np.random.randint(1, h)
                if np.random.randint(0, 2) == 0:
                    # 生成白色噪声（盐噪声）
                    result[x, y] = 0
                else:
                    # 生成黑色噪声（椒噪声）
                    result[x, y] = 255
            return result

        self.image_shop1 = self.image  # 用dolphins
        self.label.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.image_shop1)))
        self.color_image_noisy = add_noisy(self.image_shop1, 10000)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.color_image_noisy)))
     # shif检测（尺度不变特征变换）它能够检测出图像中的局部特征，并且这些特征在不同尺度和方向下保持不变，对旋转、尺度缩放、亮度变化等具有良好的鲁棒性。
    def shift_detect(self):
        sift = cv2.SIFT_create()  # 创建SIFT检测器对象
        kps = sift.detect(self.image)  # 使用SIFT检测器在self.image上检测关键点
        # 将检测到的关键点绘制到图像上，None表示没有描述符，-1表示没有掩码
        # cv2.DrawMatchesFlags_DEFAULT是绘制关键点的默认标志，cv2.DrawMatchesFlags_DEFAULT：绘制关键点的默认标志，如绘制圆圈
        self.image_sift = cv2.drawKeypoints(self.image, kps, None, -1, cv2.DrawMatchesFlags_DEFAULT)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.image_sift)))

    def image_lunkuo(self):

        kernel = np.ones((3, 3), dtype=np.uint8)
        open = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)#开运算
        close = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)#闭运算
        self.image_gradient = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)#梯度运算
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.image_gradient)))
    def image_liantongyu_detect(self):#连通域检测



        def generate_random_color():
            return np.random.randint(0, 256, 3)#用于生成一个随机颜色，返回一个包含三个随机整数（0到255之间）的数组，代表RGB颜色值。

        def fill_color(img1, n, img2):#函数用于为每个连通域填充随机颜色。它接受三个参数：img1（原图像），n（连通域的数量），img2（连通域标记图像）。
            h, w = img1.shape
            res = np.zeros((h, w, 3), img1.dtype)
            # 生成随机颜色
            random_color = {}
            for c in range(1, n):
                random_color[c] = generate_random_color()
            # 为不同的连通域填色
            for i in range(h):
                for j in range(w):
                    item = img2[i][j]
                    if item == 0:
                        pass
                    else:
                        res[i, j, :] = random_color[item]
            return res

        if __name__ == '__main__':
            # 对图像进行读取，并转换为灰度图像
            rice = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            if rice is None:
                print('..........')
                sys.exit()
            # 将图像转成二值图像
            rice_BW = cv2.threshold(rice, 50, 255, cv2.THRESH_BINARY)
            # 统计连通域
            count, dst = cv2.connectedComponents(rice_BW[1], ltype=cv2.CV_16U)#使用cv2.connectedComponents函数对二值图像rice_BW[1]进行连通域检测，返回连通域的数量count和标记图像dst。ltype=cv2.CV_16U指定输出的标记图像的数据类型为16位无符号整数。
            # 以不同颜色标记出不同的连通域
            self.result1 = fill_color(rice, count, dst)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result1)))

    def image_increase_color(self):#图像增强
        img = self.image.copy()
        alpha = 1.5  # 对比度控制1-3
        beta = 50  # 亮度控制，0-100
        enhance_color = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # 增强cv2.convertScaleAbs图像颜色
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(enhance_color)))
    def image_koutu(self):#抠图
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)#使用Canny算子对灰度图像gray进行边缘检测，低阈值设为100，高阈值设为200。
        # 创建一个空白的图像掩膜
        mask = np.zeros_like(edges)
        # edges值大于0的像素在mask掩膜中对应的位置设置为255（白色）
        mask[edges > 0] = 255
        # 对原始图像进行抠图
        self.result = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result)))
    def image_xiufu(self):
        _, mask1 = cv2.threshold(self.image, 245, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#结构元素 k
        mask1 = cv2.dilate(mask1, k)
        # cv2.inpaint 函数对原始图像 self.image 进行修复操作。5 是修复算法的半径，cv2.INPAINT_NS 是修复算法的类型，即Navier-Stokes算法，用于根据图像周围的像素信息来估计和填补损坏区域。
        self.result1 = cv2.inpaint(self.image, mask1[:, :, -1], 5, cv2.INPAINT_NS)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result1)))


    def image_diejia(self):
        #图层叠加
        under_image = self.image.copy()
        # 读取新图层图片
        new_layer = self.image2
        new_layer_resized = cv2.resize(new_layer, (100, 100))  # 调整新图层的大小为100x100像素
        x_offset = 50  # 新图层在原始图片中的x坐标
        y_offset = 50  # 新图层在原始图片中的y坐标
        # 将新图层添加到原始图片上
        under_image[y_offset:y_offset + new_layer_resized.shape[0],
        x_offset:x_offset + new_layer_resized.shape[1]] = new_layer_resized
        # 显示添加新图层后的图片
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(under_image)))
    def addtext(self):

        text=self.lineEdit_7.text()


        imageDst = np.zeros(self.image.shape, np.uint8)
        cv2.copyTo(self.image, None, imageDst)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imageDst, text, (50, 50), font, 2, (20, 200, 20), 2)
        self.image_intro = imageDst
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.image_intro)))
    def image_merge(self):
        #图像融合
        #图像融合是在图像加法的基础上增加了系数和亮度调节量，它与图像的主要区别如下：

        #图像加法：目标图像 = 图像1 + 图像2
        #图像融合：目标图像 = 图像1 × 系数1 + 图像2 × 系数2 + 亮度调节量
        try:
            # 从lineEdit_6获取透明度值
            transparency = float(self.lineEdit_6.text())
            if transparency < 0 or transparency > 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字作为透明度（0-1之间）。")
            return
        def resize_image(image, size):
            """
            这个代码中，resize_image函数接受两个参数：image（要调整尺寸的源图像）和size（一个元组，指定新的宽度和高度）。
            这个函数使用cv2.resize来调整图像尺寸，并返回调整后的图像。
            """
            return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)#interpolation 参数用于指定图像缩放时使用的插值方法。interpolation 参数用于指定图像缩放时使用的插值方法。
        def resize_image_aspect_ratio(image, target_width):#保持长宽比缩放
            orig_width, orig_height = image.shape[1], image.shape[0]
            aspect_ratio = orig_width / orig_height
            new_width, new_height = target_width, int(target_width / aspect_ratio)
            if new_height > target_width:  # 如果新高度超过目标宽度，调整宽度
                new_width, new_height = int(target_width * aspect_ratio), target_width
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        src1 = self.image
        src2 = resize_image_aspect_ratio(self.image2, src1.shape[1])  # 调整src2的尺寸以匹配src1的宽度
        # 确保高度相同
        if src1.shape[0] != src2.shape[0]:
            src2 = resize_image(src2, (src1.shape[1], src1.shape[0]))

        result = cv2.addWeighted(src1, transparency, src2, 1-transparency, 0)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(result)))

    def resize_image(self):
        # 从lineEdit_4和lineEdit_5获取缩小比例
        width_ratio_str = self.lineEdit_4.text().strip()
        height_ratio_str = self.lineEdit_5.text().strip()
        print(width_ratio_str, height_ratio_str)

        # 检查输入是否为空
        if not width_ratio_str or not height_ratio_str:
            QMessageBox.warning(self, "警告", "请输入有效的数字作为缩小比例。")
            return

        try:
            width_ratio = float(width_ratio_str)
            height_ratio = float(height_ratio_str)
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字作为缩小比例。")
            return

        if width_ratio <= 0 or height_ratio <= 0:
            QMessageBox.warning(self, "警告", "缩小比例必须大于0。")
            return

        # 计算新的尺寸
        new_width = int(self.image.shape[1] * width_ratio)
        new_height = int(self.image.shape[0] * height_ratio)

        # 调整图像尺寸
        resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(resized_image)))
    def pushbutton_caijian(self):

        try:
            # 从lineEdit_2和lineEdit_3获取裁剪比例
            width_ratio = float(self.lineEdit_2.text())
            height_ratio = float(self.lineEdit_3.text())
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字作为裁剪比例。")
            return

        if width_ratio <= 0 or height_ratio <= 0:
            QMessageBox.warning(self, "警告", "裁剪比例必须大于0。")
            return
        img1 = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

            # 计算新的尺寸
        new_width = int(img1.shape[1] * width_ratio)
        new_height = int(img1.shape[0] * height_ratio)

        # 裁剪图片
        cropped_image = img1[0:new_height, 0:new_width]  # 修正索引

        # 将OpenCV的BGR图像转换为RGB格式，以便在QLabel中显示
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(cropped_image)))
    def pushbutton_process(self):
        # 获取选中的单选按钮的ID
        id = self.button_group.checkedId()
        if id == 1:
            # 彩色图像均衡化
            img = self.image
            (b, g, r) = cv2.split(img)
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            self.result = cv2.merge((bH, gH, rH), )
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result)))
            # 执行Option 1的功能
        elif id == 2:
            imGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(imGray)))
            # 执行Option 2的功能
        elif id == 3:
            lena1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.t, self.rst = cv2.threshold(lena1, 127, 255, cv2.THRESH_BINARY)
            # 使用Format_Grayscale8确保与灰度图像兼容
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.rst)))
        elif id == 4:
            self.blur_result = cv2.blur(self.image, (5, 5))
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.blur_result)))
        elif id == 5:
            self.r = cv2.medianBlur(self.image, 3)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.r)))
        elif id == 6:
            self.r = cv2.boxFilter(self.image, -1, (2, 2), normalize=0)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.r)))
        elif id == 7:
            self.r = cv2.GaussianBlur(self.image, (5, 5), 0, 0)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.r)))
        elif id == 8:
            self.r = cv2.bilateralFilter(self.image, 25, 100, 100)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.r)))
        elif id == 9:
            o = self.image
            kernel = np.ones((5, 5), np.uint8)
            self.result1 = cv2.erode(o, kernel)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result1)))
        elif id == 10:
            o = self.image
            kernel = np.ones((9, 9), np.uint8)
            self.result1 = cv2.dilate(o, kernel)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result1)))
        elif id == 11:
            source = self.image
            ret, result = cv2.threshold(source, 127, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            blackhat = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(blackhat)))  # 黑帽
            # 黑帽。只能看见大概轮廓
            # 黑帽=闭运算-原始输入
        elif id == 12:
            # 掩模
            a = self.image
            w, h, c = a.shape
            m = np.zeros((w, h), dtype=np.uint8)
            m[100:400, 200:400] = 255
            m[100:500, 100:200] = 255
            c = cv2.bitwise_and(a, a, mask=m)
            self.result = cv2.bitwise_and(a, a, mask=m)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.result)))
    def openimage(self):

            file_name, _ = QFileDialog.getOpenFileName(self, "打开图像", QDir.currentPath(),
                                                        "Images (*.png *.bmp *.jpg *.tif)")  # 选择哪张图叠加
            if not file_name:  # 检查用户是否取消了对话框
                QMessageBox.information(self, "信息", "未选择任何文件")
                return
            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            if self.image is None:  # 检查图像是否读取成功
                QMessageBox.information(self, "错误", "无法读取图像文件")
                return
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            self.label.setPixmap(QPixmap.fromImage(ndarry2QImage(self.image)))
            self.label_2.setPixmap(QPixmap(""))

    def saveimage(self):
        result=self.label_2.pixmap()
        filepath,_ = QFileDialog.getSaveFileName(self, "保存图像", QDir.currentPath(), "Images(*.png *.bmp *.jpg *.tif）")
        if filepath:
            result.save(filepath)
            QMessageBox.information(self, "提示", "保存成功")
        else:
            QMessageBox.information(self, "提示", "保存失败")
    def show_page_1(self):
        # 切换到页面1
        self.stackedWidget.setCurrentIndex(0)
    def show_page_2(self):
        # 切换到页面2
        self.stackedWidget.setCurrentIndex(1)
    def show_quan(self):
        # 使窗体全屏显示
        self.showFullScreen()
    def exit_fullscreen(self):
        # 退出全屏模式
        self.showNormal()
    def increase_size_150(self):
        # 获取当前窗体尺寸
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.8)

        self.width = int(self.screenwidth * 0.8)

        self.resize(self.width, self.height)
        # 确保窗口居中
        self.move(self.screen().availableGeometry().center() - self.rect().center())


    def clear_all(self):
        self.label.setPixmap(QtGui.QPixmap(''))
        self.label_2.setPixmap(QtGui.QPixmap(''))

    def clear_graphics(self):
        # 获取QGraphicsScene
        scene = self.graphicsView.scene()
        # 清空场景中的所有项
        scene.clear()


    def flip_image(self, index):
        # 根据comboBox的选项执行翻转操作
        if self.image is not None:
            if index == 0:  # 垂直翻转
                return
            elif index == 1:  # 垂直翻转
                self.x = cv2.flip(self.image, 0)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.x)))
            elif index == 2:
                self.y = cv2.flip(self.image, 1)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.y)))
            else:  # X轴翻转，即先水平后垂直翻转
                self.x = cv2.flip(self.image, -1)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.x)))

    def lvjing_image(self,index):#滤镜
        if self.image is not None:
            if index == 1:
                # 水彩画滤镜
                result1 = self.image
                result = cv2.stylization(result1, sigma_s=60, sigma_r=0.6)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(result)))
            elif index == 2:
                lap_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # 拉普拉斯9的锐化
                self.dst = cv2.filter2D(self.image, cv2.CV_8U, lap_9)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.dst)))
            elif index == 3:
                ker = np.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [0, 1, 2]])
                self.img = cv2.filter2D(self.image, -1, kernel=ker)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.img)))



    def pushbutton_xuanzhuan(self):
        try:
            # 从lineEdit_angle获取旋转度数
            text = self.lineEdit.text()
            angle = int(text)
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的整数作为旋转度数。")
            return

        height, width = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.6)  # 使用输入的角度,设置中心，然后角度，后缩放比例
        self.rotate = cv2.warpAffine(self.image, M, (width, height))# cv2.warpAffine 函数执行实际的旋转操作
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(ndarry2QImage(self.rotate)))
    def open_image(self):#打开缩略图
        # 打开文件对话框选择图片
        fileName, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not fileName:  # 检查用户是否取消了对话框
            QMessageBox.information(self, "信息", "未选择任何文件")
            return
        self.image2 = cv2.imread(fileName, cv2.IMREAD_COLOR)
        if self.image is None:  # 检查图像是否读取成功
            QMessageBox.information(self, "错误", "无法读取图像文件")
            return
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        Qimage = ndarry2QImage(self.image2)
        pixmap = QPixmap.fromImage(Qimage)
        scaled_pixmap = pixmap.scaled(self.graphicsView.width(), self.graphicsView.height(), Qt.KeepAspectRatio)#并调整其大小以适应 QGraphicsView 的尺寸，
        self.scene.clear()  # 清除场景中的旧图像
        pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
        self.scene.addItem(pixmap_item)

        # 更新QGraphicsView以显示新的场景
        self.graphicsView.setScene(self.scene)
        self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        if not pixmap.isNull():
            # 如果之前有图片，先清除
            self.scene.clear()

            # 创建QGraphicsPixmapItem并添加到场景中
            pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmapItem)

            # 调整QGraphicsPixmapItem的大小以适应QGraphicsView
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            QMessageBox.warning(self, "警告", "无法加载图片，请检查文件是否有效。")

def main():
    app = QApplication(sys.argv)
    dipWindow = DIPWindow()
    dipWindow.show()
    app.exec_()

#判定执行代码是否在主模块中，如果是则调用main()函数
if __name__ == '__main__':
    main()