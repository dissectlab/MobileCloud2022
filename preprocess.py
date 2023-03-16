import cv2
import os
import time
import background_sub as bsub
from pathlib import Path
import argparse
import json
from multiprocessing import Pool
import multiprocessing

parser = argparse.ArgumentParser(description='Please specify the directory of data set')
parser.add_argument('--data_dir', type=str, default='data/originals',
                    help="the directory which contains the pictures set.")
parser.add_argument('--fg_dir', type=str, default='data/fg',
                    help="the directory which contains the foreground pictures set.")
parser.add_argument('--bg_dir', type=str, default='data/bg',
                    help="the directory which contains the background pictures set.")
parser.add_argument('--all_dir', type=str, default='data/all',
                    help="the directory which contains the original pictures set per timestamp.")
parser.add_argument('--fg_adc', type=int, default=25,
                    help="the advancement of foreground mask.")
parser.add_argument('--bg_adc', type=int, default=0,
                    help="the advancement of background mask.")

args = parser.parse_args()

# create cv2 background module for each camera
backSub = {}
for image_dir in os.listdir(args.data_dir):
    backSub[image_dir] = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=216, detectShadows=False)


def f(item):
    str_timestamp = item[1]
    image_dir = item[0]
    img_file_name = str_timestamp + ".png"
    image = cv2.imread(os.path.join(args.data_dir, image_dir, img_file_name))
    extracted_binary_foreground = backSub[image_dir].apply(image)
    # create background and foreground images
    # save foreground image
    success, img_mask, bg, box_areas = bsub.create_fg_mask(extracted_binary_foreground, image,
                                                           fg_advancement=args.fg_adc, bg_advancement=args.bg_adc,
                                                           color=True)
    # msa.append(box_areas)

    cv2.imwrite(os.path.join(fg_dir, image_dir + ".png"), img_mask)  # regular image

    # save background image
    _, img_mask, _, _ = bsub.create_fg_mask(extracted_binary_foreground, image, fg_advancement=args.fg_adc,
                                            bg_advancement=args.bg_adc, color=False)

    background_mask = bsub.create_background(img_mask, image, color=True)
    cv2.imwrite(os.path.join(bg_dir, image_dir + ".png"), background_mask)  # regular image

    # copy original images to "/all_dir"
    cv2.imwrite(os.path.join(all_dir, image_dir + ".png"), image)  # regular image
    print("\t finish image {} from camera #{}".format(img_file_name, image_dir))
    return box_areas


for timestamp in range(0, 20):

    start = time.time()

    str_timestamp = str(timestamp).zfill(5)

    print("current timestamp: ", str_timestamp, '-' * 50)
    fg_dir = os.path.join(args.fg_dir, str_timestamp)
    bg_dir = os.path.join(args.bg_dir, str_timestamp)
    all_dir = os.path.join(args.all_dir, str_timestamp)

    if timestamp > 0:
        Path(fg_dir).mkdir(parents=True, exist_ok=True)
        Path(bg_dir).mkdir(parents=True, exist_ok=True)
        Path(all_dir).mkdir(parents=True, exist_ok=True)

    if timestamp == 0:  # run background subtraction algorithm on first image
        for image_dir in ["000", "001", "002", "003", "004", "005", "006"]:
            str_timestamp = str(timestamp).zfill(5)
            img_file_name = str_timestamp + ".png"
            image = cv2.imread(os.path.join(args.data_dir, image_dir, img_file_name))
            backSub[image_dir].apply(image)

        continue
    #msa = []

    #dirs = ["000", "001", "002", "003", "004", "005", "006"]

    items = []

    images = ["000", "001", "002", "003", "004", "005", "006"]

    #if args.list is not None:
    #    for ID in args.my_list:
   #         images.remove(ID)

    for ID in images:
        items.append([ID, str_timestamp])

    with Pool(multiprocessing.cpu_count()) as p:
        print(p.map(f, items))

    #if timestamp > 0:
    #    with open(fg_dir + '/mask.json', 'w', encoding='utf-8') as f:
    #        json.dump({"msa": msa}, f, indent=4)

    print("background remove", time.time() - start)