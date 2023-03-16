import argparse
import base64
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path
import background_sub as bsub
import cv2
import numpy as np

from networking import *
from multiprocessing import Pool
from util import initial, mkdir, background_, split, update_sfm_data
from threading import Lock
from _thread import *


class Helper:
    def __init__(self, opt):
        # create cv2 background module for each camera
        self.dirs = ["000", "001", "002", "003", "004", "005", "006"] #, "006"
        fg_dir, bg_dir, all_dir, all_output_dir = mkdir("00000", args.fg_dir, args.bg_dir, args.all_dir)
        # local_dataloader("00000", 1.0, 100, args.data_dir, all_dir, dirs)
        self.back_sub = initial(self.dirs, args.data_dir, 1.0)
        self.server_channel = connect("146.95.252.27", opt.port)
        print("connected...")
        self.bg_str_timestamp = "00000"
        self.bg_mvs_process = None
        self.bg_mvs_start = None
        self.str_timestamp = "00000"
        self.fg_sfm_path = None

        start_new_thread(self.bg_mvs, (opt,))

    def start(self, args):
        while True:
            try:
                data = recv_msg(self.server_channel)
                info = json.loads(str(data.decode('utf-8')))

                str_timestamp = info["str_timestamp"]
                if info["action"] == "background&sfm":

                    if self.bg_mvs_process is not None:
                        os.kill(self.bg_mvs_process.pid, signal.SIGSTOP)
                        print("++++++++ stop current bg_mvs")

                    fg_dir, bg_dir, all_dir, all_output_dir = mkdir(str_timestamp, args.fg_dir, args.bg_dir,
                                                                    args.all_dir)
                    msa = background_(str_timestamp, info["resolution"], fg_dir, bg_dir, args.data_dir, all_dir,
                                      self.back_sub,
                                      self.dirs)

                    start_encoding = time.time()
                    fg_img = []
                    for image_dir in self.dirs:
                        img = cv2.imread(os.path.join(fg_dir, image_dir + ".png"))
                        if info["resolution"] < 1.0:
                            img = cv2.resize(img, (int(1920 * info["resolution"]), int(1080 * info["resolution"])),
                                             interpolation=cv2.INTER_LANCZOS4)
                        fg_img.append(base64.b64encode(cv2.imencode('.png', img)[1]).decode("utf-8"))

                    msg = {"msa": msa, "fg_img": fg_img}
                    msg = json.dumps(msg).encode("utf-8")

                    start_nerworking = time.time()
                    print("+ encoding {}".format(round(time.time() - start_encoding, 4)))
                    send_msg(self.server_channel, msg)
                    print("+ networking {}".format(round(time.time() - start_nerworking, 4)))

                    start_new_thread(self.sfm, (all_dir, str_timestamp, all_output_dir, bg_dir, fg_dir, msa))

                elif info["action"] == "fg_sfm":
                    with open(self.fg_sfm_path, 'rb') as file:
                        fg_sfm_ply = file.read()
                    msg = {"fg_gt_sfm_ply": base64.encodebytes(fg_sfm_ply).decode("utf-8") }
                    send_msg(self.server_channel, json.dumps(msg).encode("utf-8"))

                elif info["action"] == "update_bg_mvs":
                    while self.bg_str_timestamp == "00000":
                        pass
                    all_output_dir = os.path.join(args.all_dir, self.bg_str_timestamp + "_output")
                    with open(os.path.join(all_output_dir, "bg_mvs", "scene_dense.ply"),
                              'rb') as file:
                        bg_mvs_ply = file.read()
                    msg = {"bg_mvs_ply": base64.encodebytes(bg_mvs_ply).decode("utf-8"),
                           "bg_str_timestamp": self.bg_str_timestamp}
                    send_msg(self.server_channel, json.dumps(msg).encode("utf-8"))

            except:
                self.server_channel.close()
                print(traceback.format_exc())
                break

    def sfm(self, all_dir, str_timestamp, all_output_dir, bg_dir, fg_dir, msa):
        # start openMVG with golden configuration
        update_sfm_data(args, all_dir, del_cameras, str_timestamp, all_output_dir)
        start = time.time()
        p = subprocess.Popen(
            ["python3", args.reconstructor, all_dir, all_output_dir, "--sfm", "sfm_data.bin", "--mvs_dir",
             "mvs", "--preset", "OPENMVG"])
        p.wait()
        print("{:<60} {:>20}".format("SfM Pipeline " + all_dir + "/" + str_timestamp,
                                     '\x1b[6;30;42m' + '[finish]' + '\x1b[0m' + " in " + str(
                                         round(time.time() - start, 4))))
        time_split, fg_sfm_path, bg_sfm_path = split(all_output_dir, bg_dir, fg_dir,
                                                     np.array(msa).reshape(-1))
        self.fg_sfm_path = fg_sfm_path
        if self.bg_mvs_process is not None:
            print("++++++++ resume current bg_mvs")
            os.kill(self.bg_mvs_process.pid, signal.SIGCONT)

        self.str_timestamp = str_timestamp

    def bg_mvs(self, opt):
        while True:
            while self.str_timestamp == self.bg_str_timestamp:
                time.sleep(0.5)
                pass
            new_str_timestamp = self.str_timestamp
            start = time.time()
            all_output_dir = os.path.join(opt.all_dir, self.str_timestamp + "_output")
            bg_dir = os.path.join(opt.bg_dir, self.str_timestamp)
            self.bg_mvs_process = subprocess.Popen(
                ["python3", opt.reconstructor, bg_dir, all_output_dir, "--sfm", "bg.bin", "--mvs_dir",
                 "bg_mvs",
                 "--preset", "OPENMVS"])

            print("start bg_mvs {}, pid={}".format(new_str_timestamp, self.bg_mvs_process.pid))
            self.bg_mvs_process.wait()
            self.bg_mvs_process = None
            time_ = round(time.time() - start, 4)
            self.bg_str_timestamp = new_str_timestamp
            print("{:<60} {:>20}".format("++++++++ MVS Pipeline BG " + bg_dir, '\x1b[6;30;42m' + '[finish]' + '\x1b[0m' + " in " + str(
                                             time_)))

del_cameras = {
    "7": [],
    "6": ["000"],
    "5": ["000", "006"],
    "4": ["000", "001", "006"],
    "3": ["000", "001", "005", "006"]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please specify the directory of data set')
    parser.add_argument('--data_dir', type=str, default='data/originals',  #/media/zxj/easystore/Dance1/
                        help="the directory which contains the pictures set.")
    parser.add_argument('--fg_dir', type=str, default='data/fg',
                        help="the directory which contains the foreground pictures set.")
    parser.add_argument('--bg_dir', type=str, default='data/bg',
                        help="the directory which contains the background pictures set.")
    parser.add_argument('--all_dir', type=str, default='data/all',
                        help="the directory which contains the original pictures set per timestamp.")
    parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_dance.json', # sfm_data_global.json
                        help="the directory which contains the pictures set.")
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help="the directory which contains the final results.")
    parser.add_argument('--reconstructor', type=str, default='MvgMvsPipeline.py',
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--n', type=str, default="7",
                        help="number of cameras")
    parser.add_argument('--resolution', type=float, default=1.0,
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--port', type=int, default=8003,
                        help="the directory which contains the reconstructor python script.")

    args = parser.parse_args()
    print(args)

    h = Helper(args)

    h.start(args)


