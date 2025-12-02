<div align="center">

<h1>Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration</h1>

<div>
    <a href='https://24wenjie-li.github.io/' target='_blank'>Wenjie Li</a><sup>1</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Xiangyi_Wang2' target='_blank'>Xiangyi Wang</a><sup>1</sup>&emsp;
    <a href='https://gh-home.github.io/' target='_blank'>Heng Guo</a><sup>1</sup>&emsp;
    <a href='https://guangweigao.github.io/' target='_blank'>Guangwei Gao</a><sup>2</sup>&emsp;
    <a href='https://zhanyuma.cn/index.html' target='_blank'>Zhanyu Ma</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>Beijing University of Posts and Telecommunications&emsp; 
    <sup>2</sup>Nanjing University of Science and Technology&emsp; 
</div>

<div>
    :triangular_flag_on_post: <strong>Accepted to NeurIPS 2025</strong>
</div>

<div>
    <h4 align="center">
        • [<a href="https://24wenjie-li.github.io/projects/SSDiff/" target='_blank'>Project</a>]  &emsp; [<a href="https://arxiv.org/pdf/2510.12114" target='_blank'>arXiv</a>]  &emsp;  [<a href="" target='_blank'>Appendix</a>]•
    </h4>
</div>

<!-- <img src="assets/teaser.png" width="800px"/> -->

---
</div>

### :postbox: Update
- 2025.10.11: This repo is created.

### :wrench: TODO
- [ ] Relase restoration results.
- [ ] Relase our constructed old-photo face dataset, VintageFace.
- [x] Relase checkpoint and script for old photo face restoration.
- [x] Relase codes and config files.
- [x] Release paper on arxiv.


### Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/PRIS-CV/SSDiff
cd SSDiff

# create new anaconda env
conda create -n SSDiff python=3.8 -y
conda activate SSDiff

# install python dependencies
conda install mpi4py
pip3 install -r requirements.txt
pip install -e .
```
<!-- conda install -c conda-forge dlib -->


### Quick Inference

#### Download Pre-trained Models:
Download the facelib and dlib pretrained models from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1b_3qwrzY_kTQh0-SnBoGBgOrJ_PLZSKm?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EvDxR7FcAbZMp_MA9ouq7aQB8XTppMb3-T0uGZ_2anI2mg?e=DXsJFo)] to the `weights/facelib` folder. You can manually download the pretrained models OR download by running the following command:
```
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib (only for dlib face detector)
```


### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{li2025self,
        title={Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration},
        author={Li, Wenjie and Wang, Xiangyi and Guo, Heng and Gao, Guangwei and Ma, Zhanyu},
        booktitle={NeurIPs},
        year={2025}
    }


### Contact
If you have any questions, please feel free to reach me out at `lewj2408@gmail.com`. 