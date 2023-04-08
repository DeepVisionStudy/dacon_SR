import os
import cv2
import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='train')
    parser.add_argument("--dst", type=str, default='train_val_split')
    parser.add_argument("--size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.dst = args.dst + '_seed' + str(args.seed) + '_size' + str(args.size) # Ex. train_val_split_seed0_size0.1

    if not os.path.exists(args.dst):
        os.makedirs(os.path.join(args.dst,'train','lr'), exist_ok=True)
        os.makedirs(os.path.join(args.dst,'train','hr'), exist_ok=True)
        os.makedirs(os.path.join(args.dst,'valid','lr'), exist_ok=True)
        os.makedirs(os.path.join(args.dst,'valid','hr'), exist_ok=True)

    total = [ i for i in os.listdir(os.path.join(args.src, 'lr'))]
    
    train, valid = train_test_split(total, test_size=args.size, random_state=args.seed)

    for folder in ['lr', 'hr']:
        for img in os.listdir(os.path.join(args.src, folder)):
            if img in train:
                phase = 'train'
            elif img in valid:
                phase = 'valid'

            src_path = os.path.join(args.src, folder, img)
            src = cv2.imread(src_path, cv2.IMREAD_COLOR)
            dst_path = os.path.join(args.dst, phase, folder, img)
            cv2.imwrite(dst_path, src)
