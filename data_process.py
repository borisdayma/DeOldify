import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import os, shutil
from tqdm import tqdm
from fastai.core import parallel, defaults


def resize_and_save_data(raw_data_paths, processed_data_path, width, height):
    ''' Resize images to target size and save to directory '''
    
    print('\nResizing and Saving data')
    for raw_data_path in raw_data_paths:
        img_filenames = glob.glob(raw_data_path + "/*")
        for img_path in img_filenames:
            img_BGR = cv2.imread(img_path)
            if img_BGR.shape != (256, 256, 3):
                img_BGR = cv2.resize(img_BGR, (width, height))
                stem = Path(img_path).stem
                processed_path = processed_data_path + "/" + stem + ".jpg"
                cv2.imwrite(processed_path, img_BGR)
    print('Resizing and Saving data complete\n')

def split_train_valid_data(data_path, valid_split = 0.2):
    ''' Split data between train and valid folders '''

    img_filenames = glob.glob(data_path + "/*")
    random.shuffle(img_filenames)
    os.makedirs(data_path + "/train", exist_ok=True)
    os.makedirs(data_path + "/valid", exist_ok=True)
    split = round(len(img_filenames) * valid_split)
    for img in img_filenames[:split]:
        shutil.move(img, data_path + "/valid/")
    for img in img_filenames[split:]:
        shutil.move(img, data_path + "/train/")

def find_black_and_white(data_path, delete=False):
    ''' Return list of pictures which are in black and white '''

    print('Finding black and white images')
    if delete:
        print('Images going to be removed')
    img_b_and_w = []
    img_filenames = glob.glob(data_path + "/**/*.jpg", recursive=True)
    for img_path in tqdm(img_filenames):
        img_BGR = cv2.imread(img_path)
        # Check Cr and Cb channels are constant
        img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        Cr_var = img_YCrCb[...,1].max() - img_YCrCb[...,1].min()
        Cb_var = img_YCrCb[...,2].max() - img_YCrCb[...,2].min()
        if not (Cr_var or Cb_var):
            tqdm.write(f'{img_path} - {len(img_b_and_w)} images')
            img_b_and_w.append(img_path)
            if delete:
                os.remove(img_path)
    print('Found {} black and white images'.format(len(img_b_and_w)))
    return img_b_and_w

def parallel_find_black_and_white(img_path, index):

    delete = False
    img_BGR = cv2.imread(img_path)
    # Check Cr and Cb channels are constant
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    Cr_var = img_YCrCb[...,1].max() - img_YCrCb[...,1].min()
    Cb_var = img_YCrCb[...,2].max() - img_YCrCb[...,2].min()
    if not (Cr_var or Cb_var):
        if delete:
            os.remove(img_path)
        else:
            print(img_path)


if __name__ == "__main__":

    # data_path = '/data/Open_Images'
    # img_filenames = glob.glob(data_path + "/**/*.jpg", recursive=True)
    # parallel(parallel_find_black_and_white, img_filenames)

    find_black_and_white('/data/Open_Images', delete=True)