import os
import cv2
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='train_val_split_seed0_size0.1/split64/train/hr')
    parser.add_argument("--dst", type=str, default='HAT/hat/data/meta_info/meta_info_train_hr.txt')
    args = parser.parse_args()

    if os.path.isfile(args.dst):
        print("remove existing file...")
        os.remove(args.dst)

    with open(args.dst, "w") as file:
        for test_img in os.listdir(args.src):
            src_path = os.path.join(args.src, test_img)
            src = cv2.imread(src_path, cv2.IMREAD_COLOR)
            file.write(test_img + ' ' + str(src.shape) + '\n')
    file.close()
