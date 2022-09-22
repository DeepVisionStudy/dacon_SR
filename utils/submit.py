import os
import shutil
import zipfile
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='results_merge')
    parser.add_argument("--dst", type=str, default='results_submit')
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)
    
    os.chdir(args.src)
    
    submission = zipfile.ZipFile("submission.zip", 'w')
    for path in os.listdir():
        if path.endswith('.png'):
            submission.write(path)
    submission.close()

    shutil.move("submission.zip", "../results_submit/submission.zip")