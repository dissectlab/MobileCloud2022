import base64
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from _thread import start_new_thread
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from multiprocessing import Pool
import cv2
import numpy as np
import open3d
from open3d import VerbosityLevel, set_verbosity_level
import merge as merge_process
import jsonToply
import write_bg as wbg
import background_sub as bsub
from networking import recv_msg, send_msg

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def preprocess(items):
    fg_dir, bg_dir, image_dir, str_timestamp, image, extracted_binary_foreground, resolution = items
    # create background and foreground images
    success, fg_mask, bg, box_areas = bsub.create_fg_mask(extracted_binary_foreground, image,
                                                           fg_advancement=150, bg_advancement=0,
                                                           color=True)
    # msa.append(box_areas)
    #if resolution < 1.0:
    #    fg_mask = cv2.resize(fg_mask, (int(1920 * resolution), int(1080 * resolution)),
    #                          interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(fg_dir, image_dir + ".png"), fg_mask)  # regular image
    # save background image
    _, img_mask, _, _ = bsub.create_fg_mask(extracted_binary_foreground, image, fg_advancement=0,
                                            bg_advancement=0, color=False)

    background_mask = bsub.create_background(img_mask, image, color=True)
    #if resolution < 1.0:
   #     background_mask = cv2.resize(background_mask, (int(1920 * resolution), int(1080 * resolution)),
    #                                 interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(bg_dir, image_dir + ".png"), background_mask)  # regular image
    return box_areas


def initial(dirs, data_dir, resolution):
    back_sub = {}
    for ID in dirs:
        back_sub[ID] = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=216, detectShadows=False)
    print(">>> create new createBackgroundSubtractorMOG2 with resolution {}".format(str(resolution)))

    for image_dir in dirs:
        img_file_name = "00000.png"
        image = cv2.imread(os.path.join(data_dir, image_dir, img_file_name))
        back_sub[image_dir].apply(image)

    return back_sub


def mkdir(str_timestamp, fg_dir, bg_dir, all_dir):
    all_output_dir = os.path.join(all_dir, str_timestamp + "_output")

    fg_dir = os.path.join(fg_dir, str_timestamp)
    bg_dir = os.path.join(bg_dir, str_timestamp)
    all_dir = os.path.join(all_dir, str_timestamp)

    Path(fg_dir).mkdir(parents=True, exist_ok=True)
    Path(bg_dir).mkdir(parents=True, exist_ok=True)
    Path(all_dir).mkdir(parents=True, exist_ok=True)
    try:
        shutil.rmtree(all_output_dir)
    except:
        pass
    return fg_dir, bg_dir, all_dir, all_output_dir


def load_img(items):
    data_dir, ID, img_file_name, all_dir, back_sub = items
    image = cv2.imread(os.path.join(data_dir, ID, img_file_name))
    extracted_binary_foreground = back_sub.apply(image)
    cv2.imwrite(os.path.join(all_dir, ID + ".png"), image)
    return [image, extracted_binary_foreground]


def background_(str_timestamp, resolution, fg_dir, bg_dir, data_dir, all_dir, back_sub, dirs):
    print("current timestamp {}, resolution \x1b[6;30;42m{}\x1b[0m ".format(str_timestamp, resolution), '-' * 20)
    start = time.time()
    items = []

    for ID in dirs:
        img_file_name = str_timestamp + ".png"
        items.append((data_dir, ID, img_file_name, all_dir, back_sub[ID]))

    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        results = executor.map(load_img, items)

    imgs = []
    extracted_binary_foreground = []
    for result in results:
        imgs.append(result[0])
        extracted_binary_foreground.append(result[1])

    items = []
    for i in range(len(dirs)):
        # img_file_name = str_timestamp + ".png"
        # print("read image {}".format(os.path.join(data_dir, ID, img_file_name)))
        #image = cv2.imread(os.path.join(data_dir, ID, img_file_name))

        items.append((fg_dir, bg_dir, dirs[i], str_timestamp, imgs[i], extracted_binary_foreground[i], resolution))
        #cv2.imwrite(os.path.join(all_dir, ID + ".png"), image)  # regular image

    with Pool(7) as p:  # multiprocessing.cpu_count()
        msa = p.map(preprocess, items)

    print("{:<60} {:>20}".format("background subtraction", '\x1b[6;30;42m' + '[finish]' + '\x1b[0m' + " in " + str(
        round(time.time() - start, 4))))
    # msa = np.array(msa).reshape(-1)
    # print(msa)

    return msa


def sfm_start(items):
    action_type, args, str_timestamp, dirs, all_dir, all_output_dir, del_cameras, server_channel = items
    if action_type == "background":
        return get_fg_from_remote(args, str_timestamp, server_channel, dirs)
    elif action_type == "openMVG":
        return open_mvg(args, dirs, all_dir, all_output_dir, str_timestamp, del_cameras)


def open_mvg(args, dirs, all_dir, all_output_dir, str_timestamp, del_cameras):
    for ID in dirs:
        img_file_name = str_timestamp + ".png"
        image = cv2.imread(os.path.join(args.data_dir, ID, img_file_name))
        if args.resolution < 1.0:
            image = cv2.resize(image, (int(1920 * args.resolution), int(1080 * args.resolution)), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(all_dir, ID + ".png"), image)

    update_sfm_data(args, all_dir, del_cameras, str_timestamp, all_output_dir)
    # start to run openMvg
    # all_output_dir/
    #       mvs
    #       sfm/sfm_data.json
    start = time.time()
    p = subprocess.Popen(
        ["python3", args.reconstructor, all_dir, all_output_dir, "--sfm", "sfm_data.bin", "--mvs_dir", "mvs",
         "--preset", "OPENMVG"])
    p.wait()
    return round(time.time() - start, 2)


def get_fg_sfm_ply(server_channel, str_timestamp):
    msg = {"str_timestamp": str_timestamp, "action": "fg_sfm"}
    send_msg(server_channel, json.dumps(msg).encode("utf-8"))
    while True:
        data = recv_msg(server_channel)
        info = json.loads(str(data.decode('utf-8')))
        with open("data/gold_sfm_fg/" + str_timestamp + "_sfm_fg.ply", 'wb') as file:
            file.write(base64.b64decode(info["fg_gt_sfm_ply"]))
        break


def get_fg_from_remote(args, str_timestamp, server_channel, dirs):
    msg = {"str_timestamp": str_timestamp, "resolution": args.resolution, "action": "background&sfm"}
    send_msg(server_channel, json.dumps(msg).encode("utf-8"))
    msa = None
    while True:
        start_ = time.time()
        data = recv_msg(server_channel)
        info = json.loads(str(data.decode('utf-8')))
        # msa = np.array(info["msa"]).reshape(-1)
        msa = np.array(info["msa"]).reshape(-1)
        for i in range(len(info["fg_img"])):
            if str(i).zfill(3) in dirs:
                with open(os.path.join(args.fg_dir, str_timestamp, str(i).zfill(3) + ".png"), 'wb') as file:
                    file.write(base64.b64decode(info["fg_img"][i]))
        # start_new_thread(get_fg_sfm_ply, (server_channel, str_timestamp))
        # print(f"++++++++ decoding {round(time.time() - start_, 4)}")
        break
    return msa


def update_sfm_data(args, all_dir, del_cameras, str_timestamp, all_output_dir):
    with open(args.parameter, "r") as jsonFile:
        sfm = json.load(jsonFile)

    sfm["root_path"] = "/home/edge/3d-reconstruction/" + all_dir

    if args.resolution < 1.0:
        for item in sfm["views"]:
            item["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
            item["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)

        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = int(1920 * args.resolution / 1.118)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [int(1920 * args.resolution / 2.06),
                                                                                   int(1080 * args.resolution / 1.93)]

    for str1 in del_cameras[args.n]:
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

    Path(os.path.join(args.all_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)

    with open(all_output_dir + "/sfm/matches/sfm_data.json", 'w', encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)


def split(all_output_dir, bg_dir, fg_dir, msa):
    #############################################################################################################
    # Point-Cloud Split
    start = time.time()
    # convert sfm_data.bin to sfm_data.json
    if sys.platform.startswith('win'):
        cmd = "where"
    else:
        cmd = "which"

    ret = subprocess.run([cmd, "openMVG_main_SfMInit_ImageListing"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=True)
    OPENMVG_BIN = os.path.split(ret.stdout.decode())[0]

    pChange = subprocess.Popen(
        [os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i",
         all_output_dir + "/sfm/sfm_data.bin",
         "-o", all_output_dir + "/sfm/sfm_data.json"])
    pChange.wait()

    # separate sfm_data.json into background and foreground
    L, B, F = wbg.writeBG_FG(all_output_dir + "/sfm/sfm_data.json", bg_dir, fg_dir, msa, all_output_dir + "/sfm")

    jsonToply.py(all_output_dir + "/sfm/fg.json", all_output_dir + "/sfm/fg.ply")
    jsonToply.py(all_output_dir + "/sfm/bg.json", all_output_dir + "/sfm/bg.ply")

    # convert bg.json and fg.json back to bin files
    pChange = subprocess.Popen(
        [os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i",
         all_output_dir + "/sfm/fg.json",
         "-o", all_output_dir + "/sfm/fg.bin"])
    pChange.wait()

    pChange = subprocess.Popen(
        [os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i",
         all_output_dir + "/sfm/bg.json",
         "-o", all_output_dir + "/sfm/bg.bin"])
    pChange.wait()

    time_ = round(time.time() - start, 4)
    print(f"+ Point-Cloud Split finish in {bcolors.OKGREEN}{time_}{bcolors.ENDC}", "Total points:", L, ", BG 3D points:", B, ", FG 3D points:", F)
    return time_, all_output_dir + "/sfm/fg.ply", all_output_dir + "/sfm/bg.ply"


def mvs_start(items):
    action_type, args, str_timestamp, bg_str_timestamp, mvs_dir, all_output_dir, server_channel = items
    if action_type == "fg_mvs":
        # start to run openMvs for foreground
        start = time.time()
        p = subprocess.Popen(
            ["python3", args.reconstructor, mvs_dir, all_output_dir, "--sfm", "fg.bin", "--mvs_dir", "fg_mvs",
             "--preset", "OPENMVS"])
        p.wait()
        time_ = round(time.time() - start, 4)
        print(f"+ MVS Pipeline FG {bcolors.OKGREEN}{mvs_dir}{bcolors.ENDC} finish in {bcolors.OKGREEN}{time_}{bcolors.ENDC}")
        return time_
    elif action_type == "bg_mvs":
        start = time.time()
        msg = {"str_timestamp": str_timestamp, "action": "update_bg_mvs", "bg_str_timestamp": bg_str_timestamp}
        send_msg(server_channel, json.dumps(msg).encode("utf-8"))
        while True:
            data = recv_msg(server_channel)
            info = json.loads(str(data.decode('utf-8')))
            if bg_str_timestamp != info["bg_str_timestamp"]:
                with open("data/gold_bg/" + info["bg_str_timestamp"] + "_scene_dense.ply", 'wb') as file:
                    file.write(base64.b64decode(info["bg_mvs_ply"]))
                bg_str_timestamp = info["bg_str_timestamp"]
            time_ = round(time.time() - start, 4)
            # print("{:<60} {:>20}".format("MVS Pipeline BG " + mvs_dir,
                                         #'\x1b[6;30;42m' + '[finish]' + '\x1b[0m' + " in " + str(
                                         #    time_)))
            break
        return [time_, bg_str_timestamp]


def merge_start(args, str_timestamp, all_output_dir, bg_path):
    #############################################################################################################
    # merge
    start = time.time()
    merge_process.merge(all_output_dir + "/fg_mvs/scene_dense.ply",
                        bg_path,
                        args.output_dir + '/result_' + str_timestamp + '.ply')
    time_ = round(time.time() - start, 4)
    print(f"+ Merge {bcolors.OKGREEN}{bg_path[:-len('/scene_dense.ply')]}{bcolors.ENDC} finish in {bcolors.OKGREEN}{time_}{bcolors.ENDC}")
    return time_


def eva(str_timestamp, all_output_dir):
    set_verbosity_level(VerbosityLevel.Error)

    # sparse eva
    gt = open3d.read_point_cloud(os.path.join("data/gold_sfm_fg/", str_timestamp + "_sfm_fg.ply"))
    pr = open3d.read_point_cloud(all_output_dir + "/sfm/fg.ply")

    th = 0.01

    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)

    if len(d1) and len(d2):
        sparse_recall = float(sum(d < th for d in d2)) / float(len(d2))
        sparse_precision = float(sum(d < th for d in d1)) / float(len(d1))
        if sparse_recall + sparse_precision > 0:
            sparse_fscore = 2 * sparse_recall * sparse_precision / (sparse_recall + sparse_precision)
        else:
            sparse_fscore = 0

    return round(sparse_fscore, 4)




