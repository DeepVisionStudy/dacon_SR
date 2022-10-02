# https://github.com/sjcorreia/opencv_playground/blob/master/image_diff.py

import glob
import os.path as osp
import cv2
import imutils
import argparse
from skimage.metrics import structural_similarity

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=str, default='HAT/results/HAT_Dacon_')
    parser.add_argument("--B", type=str, default='HAT/results/HAT_Dacon_')
    parser.add_argument("--num", type=str, default='20000')
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--plot_size", type=int, default=2048)
    args = parser.parse_args()

    args.A = osp.join(args.A, 'visualization', 'test', args.num + '.png')
    args.B = glob.glob(osp.join(args.B, 'visualization', 'test', args.num + '*.png'))[0]

    imageA = cv2.imread(args.A)
    imageB = cv2.imread(args.B)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two imput images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    # show the output images
    if args.plot:
        cv2.imshow("Original (A)", cv2.resize(imageA, dsize=(args.plot_size, args.plot_size)))
        cv2.imshow("Modified (B)", cv2.resize(imageB, dsize=(args.plot_size, args.plot_size)))
        cv2.imshow("Diff", cv2.resize(diff, dsize=(args.plot_size, args.plot_size)))
        cv2.imshow("Thresh", cv2.resize(thresh, dsize=(args.plot_size, args.plot_size)))
        cv2.waitKey(0)

# python utils/show_diff.py --A HAT/results/HAT_Dacon_256_64_hv --B HAT/results/HAT_Dacon_uformer --num 20000