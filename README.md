### This repo is modified version for this [code](https://github.com/kenshohara/3D-ResNets-PyTorch), which is source code for [CVPR2018 paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf).

#### Requirements
- PyTorch: 0.2 version (not test for other version)
```bash
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl
pip3 install torchvision
```
- Python 3
- FFmpeg
```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

#### UCF101
- Download videos and train/test splits from this [site](http://crcv.ucf.edu/data/UCF101.php)
- Use FFmpeg to extract frames from raw videos
```bash
python3 datasets/video_jpg_ucf101.py ucf101_video_dir ucf101_frames_dir
```
- Generate train/test video list
```bash
python3 datasets/ucf101_get_list.py ucf101_frames_dir splits/ucf101_train01_raw.txt
python3 datasets/ucf101_get_list.py ucf101_frames_dir splits/ucf101_test01_raw.txt
```
-  finetune model from kinetics pretrain
```bash
mkdir logs; mkdir model;
bash ucf101_train.sh
```
- result

#### HMDB51
- Download videos and train/test splits from this [site](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- Use FFmpeg to extract frames from raw videos
```bash
python3 datasets/video_jpg_hmdb51.py hmdb51_video_dir hmdb51_frames_dir
```
- Generate train/test video list
```bash
python3 datasets/hmdb51_get_list.py hmdb51_frames_dir splits/hmdb51_train01_raw.txt
python3 datasets/hmdb51_get_list.py hmdb51_frames_dir splits/hmdb51_test01_raw.txt
```
- finetune model from kinetics pretrain
```bash
mkdir logs; mkdir model;
bash hmdb51_train.sh
```
- result
    - clip level: 53.45%
    - video leval: 
