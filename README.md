# Computer Vision in Monocular Depth Estimation
Final Project Repository: 

### Project Approach & Task: 
- The purport of article - Monodepth [Reference](https://arxiv.org/abs/1806.01260), introduces the use of the Self-Supervised Monocular Depth estimation technique to generate the high quality depth-from-color image to complement the LIDAR sensors. 
- Paragraphs mentioned in the paper implies the way of resolving the challenge in acquiring per-pixel ground-truth depth data at scale during the application with self-driving techniques. The authorship achieves state-of-the-art monocular depth estimation of the KITTI dataset, which performs well better than the supervised learning in ground truth depth training. However, the defect of this model is the occlusion boundary based on the pixels of the occlusion area are not visible on the way. 
- For this project, we will try to rephrase the model first and pursue differences applying loss or model architecture itself to approach the same feasible affection or better.

## About Monodepth:

### Introduction: 
- Per-pixel ground-truth depth data is challenging to acquire at scale. To overcome this limitation, self-supervised learning has emerged as a promising alternative for training models to perform monocular depth estimation. In this paper, we propose a set of improvements, which together result in both quantitatively and qualitatively improved depth maps compared to competing self-supervised methods.

## MileStone:

**Our milestone are basically divided into three essential part:**

| Target      | Approached Date |
| ----------- | ----------- |
| Familiar with Monodepth & All Theorem Deadline | 03/20/22 |
| Downloading Datasets | 04/01/22 |
| Re-implement Monodepth(Train and Test) | 04/06/22 |
| Optimization: Improve the Defects (If achievable) | 04/15/22 |
| Test and Training model by accessing KITTI | 04/25/22 |
| Prepare report and presentation | 05/03/22|


## 📝 About the technique we are tyring to approach

<div style="border: 5px solid black">
{
  Computer Vision Application by using Monocular-Depth-Estimation Algorithm
}
</div>

** **

* It is bascially the fundamental API, which we are tyring to approach by using self-supervised way to perform a depth estimation by using monocular instead of access binocular investigation.


## 🚦 Interactive Deocuments & References

### Monodepth [Reference](https://github.com/nianticlabs/monodepth2)

* Research on self-supervised monocular training usually explores increasingly complex architectures, loss functions, and image formation models, all of which have recently helped to close the gap with fully-supervised methods. The monodepth2 **(PyTorch implementation for training and testing depth estimation models)** show that a surprisingly simple model, and associated design choices, lead to superior predictions. In particular, it  propose (i) a minimum reprojection loss, designed to robustly handle occlusions, (ii) a full-resolution multi-scale sampling method that reduces visual artifacts, and (iii) an auto-masking loss to ignore training pixels that violate camera motion assumptions.

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="Picture/teaser.gif" alt="example input output gif" width="600" />
</p>

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```


## 💬 Digging into the "Monodepth-V2" VS "Monodepth" 


### 📘 Per-Pixel Minimum Reprojection Loss: 逐像素最小重投影误差损失

> Problem: Existing average together the reprojection error into each of the available source images, It can cause issues with pixels that are visible in the target image, but are not visible in some of the source images.
>
> Inituitive: The Per-Pixel Minimum Reprojection Loss can help with it rather than Average. It has been validated effectivelly improves the sharpness of occlusion boundaries, and leads to better accuracy.


**Introduction:** 

**Code explaination: (in Python - Pytorch)**

```python

```

### 📘 Auto-Masking Stationary Pixels: 自动过滤平稳像素

> Problem: When the camera is stationary or there is object motion in the scene, The monocular Depth estimation based on Self-supervised monocular training performance can suffer greatly. 
> 
> Intuitive: A simple auto-masking method that filters out pixels which do not change appearance from one frame to the next in the sequence. This has the effect of letting the network ignore objects which move at the same velocity as the camera, and even to ignore whole frames in monocular videos when the camera stops moving.

**Introduction:** 

**Code explaination: (in Python - Pytorch)**

```python

```


## Self-Construction Testing Dataset: 




## 💬 Results:

>  
