import numpy as np
import cv2

import fhog


# ffttools
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=(
        (cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!

# 实部虚部
def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]

# 复数乘法
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res

# 复数除法
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res

# 频谱中心化
def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
    assert (img.ndim == 2)
    # ndim中的dim是英文dimension维度的缩写
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] / 2, img.shape[0] / 2
    print("xh = ", xh)
    print("yh = ", yh)
    print("img = ", img.shape)
    img
    xh = int(xh)
    yh = int(yh)
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# recttools
def x2(rect): # 计算顶点坐标
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
        # cv2.BORDER_CONSTANT-固定值填充方式
    return res


# KCF tracker
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self.lambdar = 0.0001  # regularization
        self.padding = 2.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target

        if (hog):  # HOG feature
            # VOT
            self.interp_factor = 0.012  # linear interpolation factor for adaptation
            self.sigma = 0.6  # gaussian kernel bandwidth
            # TPAMI   #interp_factor = 0.02   #sigma = 0.5
            self.cell_size = 4  # HOG cell size
            self._hogfeatures = True
        else:  # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if (multiscale):
            self.template_size = 96  # template size
            self.scale_step = 1.05  # scale step for multi-scale estimation
            self.scale_weight = 0.96  # to downweight detection scores of other scales for added stability
        elif (fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])

    # 子像素峰值
    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    # 生成汉明窗
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t # 两个一维的相乘得到二维汉明窗

        if (self._hogfeatures): # 把一维hann复制成多通道
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def createGaussianPeak(self, sizey, sizex):  # 初始化时构建响应图
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def gaussianCorrelation(self, x1, x2):
        if (self._hogfeatures):
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            print("size_patch = ", self.size_patch)
            print("size_patch[0] = ", self.size_patch[0])
            print("size_patch[1] = ", self.size_patch[1])
            for i in range(self.size_patch[2]): # 多通道融合
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            # 'conjB=' is necessary!在做乘法之前取第二个输入数组的共轭.
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if (x1.ndim == 3 and x2.ndim == 3):
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif (x1.ndim == 2 and x2.ndim == 2):
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        print("11111111111111 = ", self._roi)
        # _roi = [0,0,0,0]
        # for item in self._roi:
        # print("item = ", item)

        # print("11111111111111111 ",self._roi)
        cx = self._roi[0] + self._roi[2] / 2
        # float 0123分别是左上角的坐标Xy,以及矩形的长和宽
        cy = self._roi[1] + self._roi[3] / 2  # float

        if (inithann):
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding
            print("self.padding = ", self.padding)
            print("self._roi[ = ", self._roi)

            if (self.template_size > 1): # template_size?? #把最大的边缩小到96，_scale是缩小比例
                                         # _tmpl_sz是滤波模板的大小也是裁剪下的PATCH大小
                print("padded_w = ", padded_w)
                print("padded_h = ", padded_h)
                if (padded_w >= padded_h):
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                print("self._scale = ", self._scale)
                self._tmpl_sz[0] = int(padded_w / self._scale) # ??
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if (self._hogfeatures):
                self._tmpl_sz[0] = int(
                    (self._tmpl_sz[0]) / (2 * self.cell_size)) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(
                    (self._tmpl_sz[1]) / (2 * self.cell_size)) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) / 2 * 2 # ??
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) / 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0]) # 选取从原图中扣下的图片位置大小
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE) # z是当前帧被裁剪下的搜索区域
        if (z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]): # 缩小到96
            self._tmpl_sz[0] = int(self._tmpl_sz[0])
            self._tmpl_sz[1] = int(self._tmpl_sz[1])
            print("self._tmpl_sz = ", self._tmpl_sz)
            z = cv2.resize(z, tuple(self._tmpl_sz))

        if (self._hogfeatures):
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)

            print("sizeY = ", mapp['sizeY'])
            print("sizeX = ", mapp['sizeX'])
            print("numFeatures = ", mapp['numFeatures'])
            self.size_patch = map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']])
            # size_patch为列表，保存裁剪下来的特征图的【长，宽，通道】
            self.size_patch = list(self.size_patch)
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1],
                                               self.size_patch[2])).T  # (size_patch[2], size_patch[0]*size_patch[1])
        else:
            if (z.ndim == 3 and z.shape[2] == 3): # 三维且第三维长度为3??  三通道，也就是RGB图像
                FeaturesMap = cv2.cvtColor(z,
                                           cv2.COLOR_BGR2GRAY)
                # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])
                # #np.int8  #0~255
            elif (z.ndim == 2):
                FeaturesMap = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            # 从此FeatureMap从-0.5到0.5
            self.size_patch = [z.shape[0], z.shape[1], 1] # size_patch为列表，保存裁剪下来的特征图的【长，宽，1】

        if (inithann):
            self.createHanningMats()  # createHanningMats need size_patch

        FeaturesMap = self.hann * FeaturesMap # 加汉宁（余弦）窗减少频谱泄露
        return FeaturesMap

    def detect(self, z, x): # z是_tmpl即特征的平均，x是当前帧的特征
        k = self.gaussianCorrelation(x, z)
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))
        # real()取实部  fftd()算FFT complexMultiplication()复数乘法
        # self._alphaf 相关滤波模板

        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
        # minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask)
        # 在一个数组中找到全局最小值和全局最大值
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float] 最大值对应的点

        if (pi[0] > 0 and pi[0] < res.shape[1] - 1): #??  #使用幅值做差来定位峰值的位置??
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if (pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv # 为什么还要返回pv

    def train(self, x, train_interp_factor): # train_interp_factor学习率
        k = self.gaussianCorrelation(x, x)
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)
        # alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        # _prob是初始化时的高斯响应图，相当于y

        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        # _tmpl是截取的特征的加权平均
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf
        # _alphaf是频域中相关滤波模板的加权平均

    def init(self, roi, image): # 初始化
        # self._roi = map(float, roi)
        self._roi = roi
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, 1)
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
        # _tmpl是截取的特征的加权平均
        # _prob是初始化时的高斯响应图
        # _alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1 # 修正边界
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.  # 框的中心点
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0)) # loc最小值??

        if (self.scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            if (self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif (self.scale_weight * new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 1 # -1???
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 1
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi