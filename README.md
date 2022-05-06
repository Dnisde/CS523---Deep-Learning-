# Computer Vision in Monocular Depth Estimation

## Project Approach & Task: 
- The purport of article - Monodepth [Reference](https://arxiv.org/abs/1806.01260), introduces the use of the Self-Supervised Monocular Depth estimation technique to generate the high quality depth-from-color image to complement the LIDAR sensors. 
- Paragraphs mentioned in the paper implies the way of resolving the challenge in acquiring per-pixel ground-truth depth data at scale during the application with self-driving techniques. The authorship achieves state-of-the-art monocular depth estimation of the KITTI dataset, which performs well better than the supervised learning in ground truth depth training. However, the defect of this model is the occlusion boundary based on the pixels of the occlusion area are not visible on the way. 
- For this project, we will try to rephrase the model first and pursue differences applying loss or model architecture itself to approach the same feasible affection or better.

## MileStone:

**Our milestone are basically divided into six essential part:**

| Target      | Approached Date |
| ----------- | ----------- |
| Familiar with Monodepth & All Theorem Deadline | 03/20/22 |
| Downloading Datasets | 04/01/22 |
| Re-implement Monodepth(Train and Test) | 04/06/22 |
| Re-implement Monodepth2(Train and Test) | 04/15/22 |
| Compare & Analyze their Results | 04/25/22 |
| Prepare report and presentation | 05/03/22|

## 📝 About the technique we are tyring to approach

<div style="border: 5px solid black">
Computer Vision Application by using Monocular-Depth-Estimation Algorithm
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

* Effective of Monodepth: 

<p align="center">
  <img src="Picture/mono1.gif" alt="example input output gif" width="600" />
</p>

* Effective of Monodepth2: 

<p align="center">
  <img src="Picture/mono2.gif" alt="example input output gif" width="600" />
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


### 📘 Per-Pixel Minimum Reprojection Loss

> Problem: Existing average together the reprojection error into each of the available source images, It can cause issues with pixels that are visible in the target image, but are not visible in some of the source images.
>
> Inituitive: The Per-Pixel Minimum Reprojection Loss can help with it rather than Average. It has been validated effectivelly improves the sharpness of occlusion boundaries, and leads to better accuracy.


### 📘 Auto-Masking Stationary Pixels

> Problem: When the camera is stationary or there is object motion in the scene, The monocular Depth estimation based on Self-supervised monocular training performance can suffer greatly. 
> 
> Intuitive: A simple auto-masking method that filters out pixels which do not change appearance from one frame to the next in the sequence. This has the effect of letting the network ignore objects which move at the same velocity as the camera, and even to ignore whole frames in monocular videos when the camera stops moving.

## How to reimplement it:

### Prerequisites
This code was tested with PyTorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. Other required modules:
```
torchvision
numpy
matplotlib
easydict
```
You can install the dependencies with:
```
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
For Monodepth2, we recommend to create a virtual environment with Python 3.6.6 `conda create -n monodepth2 python=3.6.6 anaconda`.

### 📘 Dataset
### KITTI
This algorithm requires stereo-pair images for training and single images for testing. KITTI dataset was used for training. It contains 38237 training samples. Raw dataset (about 175 GB) can be downloaded by running:
```
wget -i kitti_archives_to_download.txt -P ~/my/output/folder/
```
kitti_archives_to_download.txt may be found in this repo.

### 📘 Reimplement Monodepth Model

#### Training
Example of training can be find in Monodepth.ipynb notebook.

Model class from main_monodepth_pytorch.py should be initialized with following params (as easydict) for training:

- `data_dir`: path to the dataset folder
- `val_data_dir`: path to the validation dataset folder
- `model_path`: path to save the trained model
- `output_directory`: where save dispairities for tested images
- `input_height`
- `input_width`
- `model`: model for encoder (resnet18_md or resnet50_md or any torchvision version of Resnet (resnet18, resnet34 etc.)
- `pretrained`: if use a torchvision model it's possible to download weights for pretrained model
- `mode`: train or test
- `epochs`: number of epochs,
- `learning_rate`
- `batch_size`
- `adjust_lr`: apply learning rate decay or not
- `tensor_type`:'torch.cuda.FloatTensor' or 'torch.FloatTensor'
- `do_augmentation`:do data augmentation or not
- `augment_parameters`:lowest and highest values for gamma, lightness and color respectively
- `print_images`
- `print_weights`
- `input_channels`: Number of channels in input tensor (3 for RGB images)
- `num_workers`: Number of workers to use in dataloader
Optionally after initialization, we can load a pretrained model via `model.load`.

After that calling train() on Model class object starts the training process.

Also, it can be started via calling main_monodepth_pytorch.py through the terminal and feeding parameters as argparse arguments.

#### Testing
Example of training can be find in Monodepth notebook.

Model class from main_monodepth_pytorch.py should be initialized with following params (as easydict) for training:
- `data_dir`: path to the dataset folder
- `model_path`: path to save the trained model
- `pretrained`
- `output_directory`: where save dispairities for tested images
- `input_height`
- `input_width`
- `model`: model for encoder (resnet18 or resnet50)
- `mode`: train or test
- `input_channels`: Number of channels in input tensor (3 for RGB images)
- `num_workers`: Number of workers to use in dataloader
After that calling test() on Model class object starts testing process.

Also it can be started via calling main_monodepth_pytorch.py through the terminal and feeding parameters as argparse arguments.

### 📘 Reimplement Monodepth2 Model
#### Training


#### Testing


## 💬 Results:

### Monodepth


### Monodepth2

