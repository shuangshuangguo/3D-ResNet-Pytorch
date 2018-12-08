from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, dst_dir_path, file_name):
    video_file_path = os.path.join(dir_path, file_name)
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_dir_path, name)
    if not os.path.isdir(dst_directory_path):
        os.mkdir(dst_directory_path)

    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                pass
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
    cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  for file_name in os.listdir(dir_path):
    class_process(dir_path, dst_dir_path, file_name)
