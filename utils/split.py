import os
import cv2
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='test/lr')
    parser.add_argument("--dst", type=str, default='results_split')
    parser.add_argument("--dst_suffix", type=str, default='.png')
    parser.add_argument("--vertical", type=int, default=2)
    parser.add_argument("--horizontal", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)

    for test_img in os.listdir(args.src):
        src_path = os.path.join(args.src, test_img)
        src = cv2.imread(src_path, cv2.IMREAD_COLOR)

        x_size = int(512 / args.vertical)
        y_size = int(512 / args.horizontal)
        
        for x in range(args.vertical):
            for y in range(args.horizontal):
                dst_path = test_img[:-4] + '_' + str(y) + str(x) + args.dst_suffix # Ex) test_img[:-4] = 20000
                dst_path = os.path.join(args.dst, dst_path)
                splited = src[y_size*y:y_size*(y+1), x_size*x:x_size*(x+1), :]
                cv2.imwrite(dst_path, splited)