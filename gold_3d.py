import cv2
import sys
import os
import time

from open3d import VerbosityLevel, set_verbosity_level

import background_sub as bsub
import MJ_merge as merge_process
from pathlib import Path
import argparse
import subprocess
import shutil
import json
import open3d
#/home/zxj/zxj/distributed-3d-reconstruction/data/originals
#/media/zxj/easystore/Odzemok/data

parser = argparse.ArgumentParser(description='Please specify the directory of data set')
# /media/zxj/easystore/Dance1/data/originals
parser.add_argument('--data_dir', type=str, default='data/originals',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/gold_results_LEVEL_1_0.3',
                    help="the directory which contains the final results.")
parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_dance.json', #
                    help="the directory which contains the pictures set.")
parser.add_argument('--reconstructor', type=str, default='MvgMvsPipeline.py',
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('-l', '--list', nargs='+', type=str)

args = parser.parse_args()

#try:
    #shutil.rmtree(args.data_collect_dir)
    #shutil.rmtree(args.output_dir)
#except:
    #print(".....")


Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

if args.list is not None:
    my_list = [str(item) for item in args.list]
    print(my_list)

sparse_fscores = []

#timestamp = 1

for timestamp in range(1, 2):

#while True:

    str_timestamp = str(timestamp).zfill(5)
    print("current timestamp: ", str_timestamp, '-' * 50, args.resolution)


    try:
        shutil.rmtree(os.path.join(args.output_dir, str_timestamp + "_output"))
    except:
        print(".....")

    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)

    Path(collect_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    # go through each camera
    inx = 0

    dirs = ["000", "001", "002", "003", "004", "005", "006"]

    for image_dir in dirs:
        ignore = False
        if args.list is not None:
            for str1 in my_list:
                if str1 == image_dir:
                    ignore = True
                    break
        if ignore:
            print("ignore #{}".format(image_dir))
            continue
        # image file format = 000004.png
        img_file_name = str_timestamp + ".png"
        print("\tload image {} from camera #{}".format(img_file_name, image_dir))
        shutil.copy2(os.path.join(args.data_dir, image_dir, img_file_name), os.path.join(collect_dir, image_dir + ".png"))
        if args.resolution < 1.0:
            src = cv2.imread(os.path.join(collect_dir, image_dir + ".png"))
            output = cv2.resize(src, (int(1920*args.resolution), int(1080*args.resolution)), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(collect_dir, image_dir + ".png"), output)
        inx += 1

    # copy the sfm_data_gold.json to local
    with open(args.parameter, "r") as jsonFile:
        sfm = json.load(jsonFile)

    sfm["root_path"] = "/home/edge/3d-reconstruction/" + os.path.join(collect_dir)

    if args.resolution < 1.0:
        for item in sfm["views"]:
            item["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
            item["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)

        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = int(1920 * args.resolution / 1.118)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [int(1920 * args.resolution / 2.06),
                                                                                   int(1080 * args.resolution / 1.93)]

    if args.list is not None:
        for str1 in my_list:
            key = None
            for i in range(len(sfm["views"])):
                if str1 + ".png" == sfm["views"][i]["value"]["ptr_wrapper"]["data"]["filename"]:
                    key = sfm["views"][i]["key"]
                    for j in range(key + 1, len(sfm["views"])):
                        sfm["views"][j]["key"] -= 1
                        sfm["views"][j]["value"]["ptr_wrapper"]["id"] -= 1
                        sfm["views"][j]["value"]["ptr_wrapper"]["data"]["id_view"] -= 1
                        sfm["views"][j]["value"]["ptr_wrapper"]["data"]["id_pose"] -= 1
                    sfm["views"].remove(sfm["views"][i])
                    break

            if key is not None:
                for i in range(len(sfm["extrinsics"])):
                    if key == sfm["extrinsics"][i]["key"]:
                        for j in range(key + 1, len(sfm["extrinsics"])):
                            sfm["extrinsics"][j]["key"] -= 1
                        sfm["extrinsics"].remove(sfm["extrinsics"][i])
                        break

    sfm["intrinsics"][0]["value"]["polymorphic_id"] = sfm["views"][0]["value"]["ptr_wrapper"]["id"]
    sfm["intrinsics"][0]["value"]["ptr_wrapper"]["id"] = sfm["views"][-1]["value"]["ptr_wrapper"]["id"] + 1

    Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w', encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)

    # start to run openMvg + openMvs for foreground
    start = time.time()
    print("start to reconstruct {}".format(str_timestamp))
    p = subprocess.Popen(["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output")
                             , "--sfm", "sfm_data.bin", "--mvs_dir", "mvs", "--preset", "GOLD"])
    p.wait()
    if p.returncode != 0:
        break

    with open("time.json", "r") as jsonFile:
        time_file = json.load(jsonFile)

    time_file["timeList"].append({"record time": time.time(), "id": collect_dir, "resolution":args.resolution})

    with open('time.json', 'w', encoding='utf-8') as f:
        json.dump(time_file, f, indent=4)

    """
    set_verbosity_level(VerbosityLevel.Error)
    # sparse eva
    gt = open3d.read_point_cloud(os.path.join(args.output_dir, str_timestamp + "_output_th/mvs/scene_dense.ply"))
    pr = open3d.read_point_cloud(os.path.join(args.output_dir, str_timestamp + "_output/mvs/scene_dense.ply"))

    th = 0.005

    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)

    if len(d1) and len(d2):
        sparse_recall = float(sum(d < th for d in d2)) / float(len(d2))
        sparse_precision = float(sum(d < th for d in d1)) / float(len(d1))
        if sparse_recall + sparse_precision > 0:
            sparse_fscore = 2 * sparse_recall * sparse_precision / (sparse_recall + sparse_precision)
        else:
            sparse_fscore = 0

    sparse_fscores.append(sparse_fscore)
    """
    #args.resolution -= 0.1

    #if args.resolution <= 1.0:
    #    break

#print("F=", sparse_fscores)
