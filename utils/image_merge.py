import os
import cv2
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='HAT/results/HAT_Dacon/visualization/test')
    parser.add_argument("--dst", type=str, default='results_merge')
    parser.add_argument("--dst_size", type=int, default=2048)
    parser.add_argument("--vertical", type=int, default=2)
    parser.add_argument("--horizontal", type=int, default=2)
    args = parser.parse_args()

    args.src_suffix = '_' + args.src.split('/')[2] + '.png'

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)

    for img_num in range(20000, 20018):
        output = np.zeros((args.dst_size,args.dst_size,3), np.uint8)

        # Same method with split.py
        for x in range(args.vertical):
            for y in range(args.horizontal):
                src_path = str(img_num) + '_' + str(y) + str(x) + args.src_suffix
                src_path = os.path.join(args.src, src_path)
                src = cv2.imread(src_path, cv2.IMREAD_COLOR)

                y_size, x_size, _ = src.shape
                output[y_size*y:y_size*(y+1), x_size*x:x_size*(x+1), :] = src

        dst_path = os.path.join(args.dst, str(img_num) + '.png')
        cv2.imwrite(dst_path, output)