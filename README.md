# SP-KAN: Sparse-sine Perception Kolmogorov–Arnold Networks for Infrared Small Target Detection [[Paper]](https://www.sciencedirect.com/science/article/pii/S0924271626000705) [[Weight]](https://drive.google.com/file/d/1LF0TRmxJ0J8zm8IiDG37LH_7_9dLuFVp/view?usp=sharing) 

#### Shuai Yuan, Yu Liu, Xiaopei Zhang, Xiang Yan, Hanlin Qin, Naveed Akhtar, ISPRS Journal of Photogrammetry and Remote Sensing 2026.

### If the implementation of this repo is helpful to you, just star it！ ⭐⭐⭐

# Structure
![Image text](https://github.com/xdFai/SP-KAN/blob/main/Fig/KAN01.png)

![Image text](https://github.com/xdFai/SP-KAN/blob/main/Fig/KAN02.png)


# Introduction

We present a Sparse-sine Perception Kolmogorov–Arnold Networks (SP-KAN) to the IRSTD task. Experiments on public datasets demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose SP-KAN, which reformulate IRSTD as a global context modulation problem driven by sparse nonlinear modules.

2. We design a pattern complementarity module (PCM) to capture unstructured dependencies and local saliency interactions, enhancing target–background separability.

3. We devise a sparse-sine perception Kolmogorov–Arnold layer (SPKAL) to unlock the nonlinear representational potential of the KAN layer. 


## Usage

### Research innovation branch: target-aware boundary deep supervision

The paper (Section 4.1, Eq. (7)) trains every deep-supervision head with
unweighted BCE. Its limitations section (Section 5.9, p. 16) specifically
identifies incomplete contours and false alarms as remaining failure modes and
proposes explicit edge representations as future work. This branch implements
that follow-up idea as a reproducible hypothesis:

* `TargetAwareBoundaryLoss` estimates the foreground/background ratio per batch
  and caps the positive-pixel weight, avoiding domination by the background.
* A 3x3 morphological gradient emphasizes target boundaries, while a soft Dice
  term stabilizes training for one- or few-pixel targets.
* Deep-supervision heads use normalized weights `(0.5, 0.5, 0.75, 1, 1, 1)` so
  the loss scale is independent of the number of heads.

The original BCE objective remains available with `--loss_name bce` for a
direct ablation. Run the innovation branch with:

Additional infrared-specific module, loss, robustness, and calibration ideas
are catalogued in [INNOVATION_CATALOG.md](INNOVATION_CATALOG.md). The verified
top-venue abstract scan and module-to-code integration map are available in
[LITERATURE_STITCHING_REPORT.md](LITERATURE_STITCHING_REPORT.md).

```bash
git switch innovation/target-aware-boundary-loss
bash scripts/run_target_aware.sh --epochs 1000
```

For a fast smoke test (without requiring the full dataset), use the same
script with `--epochs 1 --max_train_steps 1 --no-auto_test` after placing a
small compatible dataset under `datasets/`. Tune the three loss controls with
`--loss_boundary_weight`, `--loss_dice_weight`, and `--loss_max_pos_weight`.

#### 1. Data

The **SIRST3** dataset, which combines **IRSTD-1K**, **NUDT-SIRST**, and **SIRST-v1**, is used to train SCTransNet.
* **SIRST-v1** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)


* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── SIRSTv1 (~which is misnamed as NUAA-SIRST~)
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUAA-SIRST.txt
  │    │    │    ├── test_NUAA-SIRST.txt
  │    ├── SIRST3 (~The sum of SIRSTv1, NUDT-SIRST and IRSTD-1K~)
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST3.txt
  │    │    │    ├── test_SIRST3.txt
  
  ```


#### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.
权重文件的百度网盘链接：https://pan.baidu.com/s/1lzdUAQpw4oLSktdxbd-tRQ?pwd=6666

权重文件的谷歌云盘链接：https://drive.google.com/file/d/1LF0TRmxJ0J8zm8IiDG37LH_7_9dLuFVp/view?usp=sharing
```bash
python test.py
```

## Results and Trained Models

#### Qualitative Results
![Image text](https://github.com/xdFai/SP-KAN/blob/main/Fig/KAN14.png)



#### Quantitative Results on  SIRSTv1, NUDT-SIRST, IRSTD-1K， and SIRST3. i.e, one weight for four Datasets.

| Model         | mIoU (x10(-2)) |  F-measure (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| SIRST v1      | 79.15  |  88.36 | 97.72 | 14.82 |
| NUDT-SIRST    | 95.24  |  97.56 | 99.36 | 3.17  | 
| IRSTD-1K      | 68.02  |  80.96 | 93.27 | 10.12 |
| SIRST3        | 84.20  |  91.42 | 98.14 | 8.01 |



*This code is highly borrowed from [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). Thanks to Xinyi Ying.

*The overall repository style is highly borrowed from [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@article{SP-KAN,
title = {SP-KAN: Sparse-sine perception Kolmogorov–Arnold networks for infrared small target detection},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {234},
pages = {1-19},
year = {2026},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2026.02.019},
url = {https://www.sciencedirect.com/science/article/pii/S0924271626000705},
author = {Shuai Yuan and Yu Liu and Xiaopei Zhang and Xiang Yan and Hanlin Qin and Naveed Akhtar},
}
```


## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](shuaiyuan@hfut.edu.cn) for any question regarding our SP-KAN.**






