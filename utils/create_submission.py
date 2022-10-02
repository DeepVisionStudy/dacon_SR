import os
import shutil
import zipfile
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='HAT/results/HAT_Dacon_AdamW/visualization/test')
    parser.add_argument("--dst", type=str, default='results_submit')
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)
    
    memory_dir = os.getcwd()
    os.chdir(args.src) # for using zipfile
    
    submission = zipfile.ZipFile("submission.zip", 'w')
    for path in os.listdir():
        if path.endswith('.png'):
            new_path = path.split('_')[0].split('.')[0] + '.png' # new_path : delete basicsr suffix in path
            shutil.move(path, new_path)
            submission.write(new_path)
    submission.close()
    
    new_path_zip = os.path.join(memory_dir, args.dst, "submission.zip")
    shutil.move("submission.zip", new_path_zip)