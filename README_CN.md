[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Rethinking Personalized Aesthetics Assessment: Employing Physique Aesthetics Assessment as An Exemplification
</b>
</h1>
<h4>
<b>
Haobin Zhong, Shuai He, Anlong Ming, Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

-----------------------------------------


## 介绍
个性化美学评估（PAA）旨在精准预测个体对美学的独特感知。随着定制化需求的不断增长，PAA使得应用能够根据个体的美学偏好，生成量身定制的结果。现有的PAA框架包括预训练和微调两个阶段，但在实际应用中面临三大挑战：

1. 模型通过通用美学评估（GAA）数据集进行预训练，但GAA的集体偏好常常导致个性化美学预测的冲突。
2. 个性化调查的范围和阶段不仅与用户相关，还与评估对象的特性息息相关；然而，目前的个性化调查方法未能充分考虑评估对象的特性。
3. 在实际应用中，个体的累积多模态反馈具有极高的价值，应被纳入模型优化过程中，但遗憾的是，这些反馈未能得到足够的关注。

为了解决上述问题，我们提出了一个全新的PAA+框架，该框架分为预训练、微调和持续学习三个阶段。为了更好地反映个体差异，我们采用了形体美学评估（PhysiqueAA）这一直观且易于理解的应用场景来验证PAA+。我们还发布了一个名为PhysiqueAA50K的数据集，包含超过50,000张注释形体图像。同时，我们开发了形体美学框架（PhysiqueFrame），并通过大规模基准测试，取得了领先的表现（SOTA）。我们的研究旨在为PAA领域提供一条创新的路径，并推动其应用发展。

<img src="paradigm_1.jpg">

## 代码使用说明


* ### **模型输出**
模型输出是预测形体美学评分，包含三个维度：外观、健康和姿势。

* ### **环境安装**
```
conda create -n physiqueAA python=3.10.14
conda activate physiqueAA
pip install -r requirements.txt
cd ./code
bash script.sh
```
从[Baidu Netdisk](https://pan.baidu.com/s/1vno-V5VoozFhLxrfkjLHqg?pwd=jx37) 下载 SMPLer_X checkpoints，并将其权重放于 `./code/SMPLer_X/pretrained_models`。

从[Baidu Netdisk](https://pan.baidu.com/s/1vno-V5VoozFhLxrfkjLHqg?pwd=jx37) 下载 smplx.npz，并将其权重放于 `.code/SMPLer_X/common/utils_smpler_x/human_model_files/smplx`。

从[Baidu Netdisk](https://pan.baidu.com/s/10KRxE95g9WnoitJ-hoO38A?pwd=6zd5) 下载 Swinv2 checkpoints，并将其权重放于 `./code/models_/pam/pretrained`。

所有资源也可以从[OneDrive](https://bupteducn-my.sharepoint.com/:f:/g/personal/hs19951021_bupt_edu_cn/EugRql7EAD1Fr6cfo0_0X-QBPILI1QAAPYzPvxKc8GlbkQ?e=TG3C0x)下载

如果遇到以下错误消息：
```
RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
```
解决方案: 在torchgeometry/core/conversions.py中更改 `1- mask` 为 `~mask `。

例如: 更改 `mask_c2 = (1 - mask_d2) * mask_d0_nd1` 为 `mask_c2 = (~mask_d2) * mask_d0_nd1`。

* ### **训练步骤**
1. 下载数据集 [Baidu Netdisk](https://pan.baidu.com/s/1NgBbu6Jf4IxrynigqO028g?pwd=kvev)。
2. 运行 `train.py` 训练网络。

* ### **推理**
1. 运行 `inference.py` 进行推理。

* ### **PhysiqueFrame 权重文件**
1. 从 [Baidu Netdisk](https://pan.baidu.com/s/1OOt2X30qe93HmW8XJbPbaQ?pwd=n124) 下载PhysiqueFrame 权重文件。

由于在企业合作项目中使用的原因，PENet无法开源。我们在数据集中提供了PENet生成的偏好特征（preference tensor）。

## 如果你觉得我们的工作对你有帮助，欢迎引用:
```
@inproceedings{zhong2025rethinking,
  title={Rethinking Personalized Aesthetics Assessment: Employing Physique Aesthetics Assessment as An Exemplification},
  author={Zhong, Haobin and He, Shuai and Ming, Anlong and Ma, Huadong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2935--2944},
  year={2025}
}
```

<table>
  <thead align="center">
    <tr>
      <td><b>🎁 Projects</b></td>
      <td><b>📚 Publication</b></td>
      <td><b>🌈 Content</b></td>
      <td><b>⭐ Stars</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Attacker-against-image-aesthetics-assessment-model"><b>Attacker Against IAA Model【美学模型的攻击和安全评估框架】</b></a></td>
      <td><b>TIP 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Attacker-against-image-aesthetics-assessment-model?style=flat-square&labelColor=343b41"/></td>
    </tr
    <tr>
      <td><a href="https://github.com/woshidandan/Rethinking-Personalized-Aesthetics-Assessment"><b>Personalized Aesthetics Assessment【个性化美学评估新范式】</b></a></td>
      <td><b>CVPR 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Rethinking-Personalized-Aesthetics-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><b>Pixel-level image exposure assessment【首个像素级曝光评估】</b></a></td>
      <td><b>NIPS 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><b>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</b></a></td>
      <td><b>ICML 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><b>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</b></a></td>
      <td><b>Information Fusion 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><b>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</b></a></td>
      <td><b>ACMMM 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><b>Image color aesthetics assessment【首个色彩美学评估】</b></a></td>
      <td><b>ICCV 2023</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><b>Image aesthetics assessment【通用美学评估】</b></a></td>
      <td><b>ACMMM 2023</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><b>Theme-oriented image aesthetics assessment【首个多主题美学评估】</b></a></td>
      <td><b>IJCAI 2022</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><b>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><b>Motion rhythm synchronization with beats【动作与韵律对齐】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><b>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</b></a></td>
      <td><b>CVPRW NTIRE 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
  </tbody>
</table>


