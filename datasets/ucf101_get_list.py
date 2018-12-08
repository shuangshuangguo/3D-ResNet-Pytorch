import re
import os
import sys

if __name__ == '__main__':
    frame_path = sys.argv[1]
    list_path = sys.argv[2]
    result_path = open('_'.join((list_path.split('_')[:-1]))+'.txt', 'w')
    with open(list_path, 'r') as f:
        for line in f:
            lines = re.split(' |\.|/|\n|\r|', line)
            print(lines)
            if 'HandStand' in lines[1]:
                file_name = 'v_Hands' + lines[1][7:]
            else:
                file_name = lines[1]
            file_path = os.path.join(frame_path, file_name)
            num_frames = len(os.listdir(file_path))
            label = int(lines[3]) - 1
            result_path.write(file_path+' '+str(num_frames) + ' ' +str(label) + '\n')
    result_path.close()
