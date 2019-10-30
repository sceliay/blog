---
categories: Notes
title: Quality assessment
date: 2019-10-13 14:58:36
tags: [image quality, video quality, audio quality]
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Image Quality Assessment(IQA)
1. 有参考图图像的质量评估
- MSE: 计算图像的像素差的平方，然后在全图上求平均：
$$ MSE=\frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}{n-1}[I(i,j) - K(i,j)]^2 $$
但是，基于 MSE 的损失不足以表达人的视觉系统对图片的直观感受。

- PSNR: 峰值信噪比(Peak-signal to Noise Ratio)
$$ PSNR=10 \times \log_{10}\frac{MAXI^2}{MSE} $$
MAXI表示颜色的最大数值。

- [SSIM(Structural Similarity)](https://zhuanlan.zhihu.com/p/67199699): 根据图像的亮度(luminance)，对比度(contrast)和结构(structure)进行相似度比较：
	- 亮度：
	$$ {\mu_x} = \frac{1}{N}\sum_{i=1}^{N}x_i $$
	$$ l(x,y) = \frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1} $$
	$$ C_1 = (K_1L)^2 $$
	其中K<<1是一个常数，常取值为0.01，L是灰度图的动态范围，由图像的数据类型决定，如果数据为 uint8 型，则 L=255。可以看出，l(x,y)对称且始终小于等于1，当 x = y 时为1。

	- 对比度：
	$$ \sigma_x = (\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\mu_x)^2)^{\frac{1}{2}} $$
	$$ c(x,y) = \frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2} $$
	$$ C_2 = (k_2L)^2 $$
	K_2 常取值为0.03，c(x,y)对称且小于等于1，当x=y时等号成立。

	- 结构相似度： 归一化的两个向量的相似度比较 

- 基于信息论基础：信息保真度准则（Information Fidelity Criterion,IFC）和视觉信息保真度（Visual Information Fidelity,VIF）

3. 无参考图图像的质量评估
- [UCIQE](https://zhuanlan.zhihu.com/p/40746930): 色彩浓度、饱和度、对比度的线性组合：
$$ UCIQE = c_1\*\sigma_c+c_x\*con_1+c_3\*\mu_s $$

- UIQM (underater image quality measurement)

- [图像统计特性](https://baike.baidu.com/item/IQA/19453034)
	- 均值： 图像像素的平均值，与亮度算法一致
	- 标准差：


# Video Quality Assessment(VQA)
参考：[视频质量评价](https://blog.csdn.net/leixiaohua1020/article/details/16359465), [VMAF](https://github.com/Netflix/vmaf)
1. 视频编码器

2. 编码标准

3. 视频质量评价
- 视频主观质量评价(Subjective Quality Assessment, SQA)
	DSIS, DSCQS, SSM
- 视频客观质量评价(Objective Quality Assessment, OQA)
	全参考(FullReference，FR)，部分参考(ReducedReference，RR)和无参考(No Reference，NR)
扩展：[视音频数据处理](https://blog.csdn.net/leixiaohua1020/article/details/50534150#comments)

4. [Nefix对视频源特性的分析](https://www.jianshu.com/p/b97e4d15a400)：
- 压缩失真，画质损失，随机噪声，几何形变
- 视频内容
- 源素材特征，如：胶片颗粒、传感器噪声、计算机生成的材质、亮度、对比度、颜色变化、色泽浓郁度、锐度。
- [VMAF](https://www.infoq.cn/article/a-quality-assessment-tool-for-video-streaming-media): 视觉信息保真度(VIF)，细节丢失指标(DLM)，运动

5. [爱奇艺短视频质量评估模型](https://www.infoq.cn/article/GfEC9QRjRgofdA7sR_H2)
- 封面图质量：模糊，黑边，拉伸变形，画面暗，无主体，无意义等。
- 视频内容质量：视频无意义，无聊，不清晰，花屏，广告，低俗等。
- 文本质量：标题过于简单，特殊符号多，句子不通顺，语法结构不正常，标题党，图文不符等。

- 封面图质量模型：基于卷积模型提取的深度特征和人工设计特征的图像质量模型。
	- 基础质量特征
		- 边缘的空间分布：将图像进行拉普拉斯滤波与其类别拉普拉斯图像均值的 L1 距离进行度量。
		- 颜色分布、色调计数、对比度与亮度：基于图像的RGB或HSV颜色空间来统计。
		- 模糊程度：模糊核算法用以评估图像或图像像素的锐度或聚焦程度。6组模糊特征，及其统计均值、方差、最大值、最小值，考虑到局部模糊性，划分图像的四个区域。
			- 基于梯度 (Gradient-based operators)，该算法假设清晰图像相比模糊图像有更锐利的线条；
			- 基于拉普拉斯变换 (Laplacian-based operators)，统计图像中线条的占比；
			- 其他包括基于小波算子 (Wavelet-based operator)、基于统计算子 (Statistic-based operators)、基于离散余弦算子 (Discrete cosine transform)、基于局部表示和滤波相结合 (Miscellaneous operators)。
	- Deep & Wide： Google NIMA深度美感模型
- 视频内容质量模型：端到端训练的基于多模态的深度内容质量模型。
	- 视频抽帧表示、光流表示和音频表示
- 文本质量模型：基于文本结构特征和文本语义特征的文本质量分类模型。
	- 语义抽取和句法结构抽取
	- 词性、依存关系、通顺度、长度、异常字符占比、类型标签

- NetVlad：视觉表示
- TSN：运动表示
- VGGISH: 音频特征
- XGBOOST：文本特征

6. MOVIE（MOtion-based Video IntegrityEvalution）：计算视频中物体的运动矢量，联合时域和空域的失真信息，最终得到一个符合主观感受的失真评价分数。