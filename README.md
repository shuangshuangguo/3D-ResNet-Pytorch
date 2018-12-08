#### This repo is modified version for this [code](https://github.com/kenshohara/3D-ResNets-PyTorch), which is source code for [CVPR2018 paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf).

##### Requirements
- PyTorch: 0.2 version (not test for other version)
```python
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl
pip3 install torchvision
```
- Python 3
- FFmpeg
```python
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

##### Prepare for the dataset
- UCF101
    - Download videos and train/test splits from this [site](http://crcv.ucf.edu/data/UCF101.php)
    - Use FFmpeg to extract frames from raw videos
    - Generate train/test video list
