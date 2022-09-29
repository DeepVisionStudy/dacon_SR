import os
import cv2
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='test/lr')
    parser.add_argument("--src_size", type=int, default=512)
    parser.add_argument("--dst", type=str, default='results_rescale')
    parser.add_argument("--dst_size", type=int, default=2048)
    parser.add_argument("-i", "--interpolation", type=str, default='lanczos4')
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)

    dsize = (args.dst_size, args.dst_size)
    
    if args.dst_size > args.src_size:
        if args.interpolation == 'lanczos4':
            interpolation = cv2.INTER_LANCZOS4
        elif args.interpolation == 'cubic':
            interpolation = cv2.INTER_CUBIC
        elif args.interpolation == 'linear':
            interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    # test_img = src name = dst name
    for test_img in os.listdir(args.src):
        src_path = os.path.join(args.src, test_img)
        src = cv2.imread(src_path, cv2.IMREAD_COLOR)

        dst = cv2.resize(src, dsize=dsize, interpolation=interpolation)
        dst_path = os.path.join(args.dst, test_img)
        cv2.imwrite(dst_path, dst)
