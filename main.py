from concurrent.futures import ThreadPoolExecutor
import time
import argparse
import json
import numpy as np
from util import sfm_start, mkdir, split, mvs_start, merge_start, bcolors, eva, get_fg_sfm_ply
from networking import server_discovery

cameras = {
        "7": ["000", "001", "002", "003", "004", "005"], # , "006"
        "6": ["001", "002", "003", "004", "005", "006"],
        "5": ["001", "002", "003", "004", "005"],
        "4": ["002", "003", "004", "005"],
        "3": ["002", "003", "004"]
    }

del_cameras = {
    "7": [],
    "6": ["000"],
    "5": ["000", "006"],
    "4": ["000", "001", "006"],
    "3": ["000", "001", "005", "006"]
}

# /media/zxj/easystore/Dance1/data/originals
# /media/zxj/easystore/Odzemok/data
parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='data/originals', #/home/zxj/zxj/distributed-3d-reconstruction/data/originals
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
parser.add_argument('--bg', type=str, default='data/gold_bg/scene_dense.ply',
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--target_bg', type=str, default="gold",
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--fg_adc', type=int, default=150,
                    help="the advancement of foreground mask.")
parser.add_argument('--bg_adc', type=int, default=0,
                    help="the advancement of background mask.")
parser.add_argument('--opt', type=int, default=0,
                    help="the advancement of foreground mask.")
parser.add_argument('--deadline', type=float, default=5.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--n', type=str, default="7",
                    help="number of cameras")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")


args = parser.parse_args()


fail = {
    "timestamp": []
}

solution = {}

deadline = args.deadline
r_min = 0.3
r_max = 1.0
r_opt = 0


minimal_check = True
maximal_check = False


hist = {
    "scale": [],
    "camera": [],
    "fscore": [],
    "time": [],
    "dense_fscore": [],
    "sparse_fscore": []
}

server_channel = server_discovery(port=8003)
bg_str_timestamp = "00000"

total_time = []
sfm_time   = []
mvs_time   = []
merge_time = []
splt_time  = []
f_score    = []
res_       = []

for i in range(1, 50):

    if i == 49 or i == 81:
        continue

    if args.opt != 1 or i == 1:
        args.resolution = round(args.resolution, 2)

    if args.opt == 1 and i == 2:
        args.resolution = r_min

    print(f"########################## (resolution = {args.resolution}, number of views = {args.n}, deadline = {args.deadline} ##########################")
    if args.opt == 1:
        print(json.dumps(solution, indent=4))
    #else:
       # print(json.dumps(solution[args.n], indent=4))

    #Path(args.output_dir + "_" + str(args.resolution)).mkdir(parents=True, exist_ok=True)

    str_timestamp = str(i).zfill(5)
    fg_dir, bg_dir, all_dir, all_output_dir = mkdir(str_timestamp, args.fg_dir, args.bg_dir, args.all_dir)

    items = [
        ("background", args, str_timestamp, cameras[str(args.n)], None, None, None, server_channel),
        ("openMVG", args, str_timestamp, cameras[str(args.n)], all_dir, all_output_dir, del_cameras, None)
    ]

    time_ = time.time()

    with ThreadPoolExecutor(2) as executor:
        results = executor.map(sfm_start, items)

    msa, local_sfm = results

    time_sfm = round(time.time() - time_, 2)
    print(f"\t[++++++++ local sfm finished in {local_sfm}]")
    print(
        f"+ SfM Pipeline {bcolors.OKGREEN}{all_dir}{bcolors.ENDC} finished in {bcolors.OKGREEN}{time_sfm}{bcolors.ENDC}")

    # print("+", list(msa))

    if args.resolution < 1.0:
        for j in range(len(msa)):
            msa[j] = round(msa[j] * args.resolution)

    time_split, fg_sfm_ply, bg_sfm_ply = split(all_output_dir, bg_dir, fg_dir, msa)

    items = [("fg_mvs", args, str_timestamp, None, fg_dir, all_output_dir, None),
             ("bg_mvs", args, str_timestamp, bg_str_timestamp, bg_dir, all_output_dir, server_channel)]

    with ThreadPoolExecutor(2) as executor:
        results = executor.map(mvs_start, items)

    time_mvs_fg, mvs_bg = results
    time_mvs = round(np.max([time_mvs_fg, mvs_bg[0]]), 4)
    bg_str_timestamp = mvs_bg[1]
    print(f"\t[++++++++ MVS Pipeline in {bcolors.OKGREEN}{time_mvs}{bcolors.ENDC}]" )

    items = [("bg_mvs", args, str_timestamp, bg_str_timestamp, bg_dir, all_output_dir, server_channel)]

    with ThreadPoolExecutor(1) as executor:
        results = executor.map(mvs_start, items)

    for result in results:
        bg_str_timestamp = result[1]

    print(f"\t[++++++++ use background {bcolors.HEADER}{bg_str_timestamp}{bcolors.ENDC}]")
    time_merge = merge_start(args, str_timestamp, all_output_dir,
                             bg_path="data/gold_bg/" + bg_str_timestamp + "_scene_dense.ply")

    get_fg_sfm_ply(server_channel, str_timestamp)

    f_score.append(eva(str_timestamp, all_output_dir))

    total_time.append(round(time_sfm + time_split + time_mvs + time_merge, 4))
    sfm_time.append(round(time_sfm, 4))
    mvs_time.append(round(time_mvs, 4))
    splt_time.append(round(time_split, 4))
    merge_time.append(round(time_merge, 4))
    res_.append(args.resolution)

    print("")
    print(f"+ SfM time = {bcolors.OKGREEN}{sfm_time}{bcolors.ENDC}")
    print(f"+ MVS time = {bcolors.OKGREEN}{mvs_time}{bcolors.ENDC}")
    print(f"+ total time = {bcolors.OKGREEN}{total_time}{bcolors.ENDC}")
    print(f"+ avg total time = {bcolors.OKGREEN}{np.average(total_time)}{bcolors.ENDC}")
    print(f"+ res = {bcolors.OKGREEN}{res_}{bcolors.ENDC}")
    print(f"+ avg res = {bcolors.OKGREEN}{np.average(res_)}{bcolors.ENDC}")
    print(f"+ F-score = {bcolors.OKGREEN}{f_score}{bcolors.ENDC}")
    # print("############################################################################################")


    # if args.resolution < 1.0:
    #    continue

    """
    while True:
        try:
            p = subprocess.Popen(
                ["cp", all_output_dir + "/sfm/fg.ply",
                 "/home/zxj/zxj/3d-reconstruction/data/sfm_eva_fg/" + str_timestamp + "_fg_960.ply"])
            p.wait()

            if p.returncode != 0:
                break

            p = subprocess.Popen(
                ["cp", all_output_dir + "/fg_mvs/scene_dense_mesh_refine_texture.ply",
                 "/home/zxj/zxj/3d-reconstruction/data/mvs_eva_fg/" + str_timestamp + "_fg_960.ply"])
            p.wait()
            if p.returncode != 0:
                break
                
            break
        except Exception as e:
            continue
    """
    continue

    if i == 1:
        continue

    if args.opt == 0:
        #window = min(len(total_time), 10)
        if deadline - np.average(total_time) > 0:
            if deadline - total_time[-1] > 0:
                args.resolution = min(round(args.resolution + 0.02, 2), 1.0)
                print(f"+ minor add 0.1 to resolution")
        elif np.average(total_time) - deadline > 0:
            if total_time[-1] - deadline > 0:
                args.resolution = max(round(args.resolution - 0.02, 2), 0.3)
                print("+ minor sub 0.1 to resolution")
        continue

    if args.n not in solution:
        solution[args.n] = {}

    next_number_camera = False

    if maximal_check:
        F_opt = 0
        R_opt = 1.0
        n_opt = "7"
        for key, value in solution.items():
            if "F-score" in value and value["F-score"] > F_opt:
                R_opt = round(value["resolution"], 2)
                F_opt = value["F-score"]
                n_opt = key
        if F_opt - f_score[-1] >= 0.03 and args.n != "7":
            args.opt = 0
            args.resolution = round(R_opt, 2)
            args.n = n_opt
            continue
        else:
            print("+ hist OPT F-score={}, resolution={}, cameras={}".format(F_opt, R_opt, n_opt))
            maximal_check = False

    if deadline > total_time[-1] or deadline + 1.0 > total_time[-1]:
        r_min = args.resolution
        r_opt = args.resolution
        if minimal_check:
            maximal_check = True
        minimal_check = False
        solution[args.n] = {"time": total_time[-1], "resolution": r_opt, "F-score": f_score[-1]}
        print(f"+ add solution {solution[args.n]}")
    else:
        if minimal_check:
            next_number_camera = True
        r_max = args.resolution
        F_opt = 0
        R_opt = 1.0
        n_opt = "7"
        for key, value in solution.items():
            if "F-score" in value and value["F-score"] > F_opt:
                R_opt = round(value["resolution"], 2)
                F_opt = value["F-score"]
                n_opt = key
        if F_opt - f_score[-1] >= 0.02 and args.n != "7":
            next_number_camera = True

    if round(r_max - r_min, 2) <= 0.02 or next_number_camera:
        if r_opt != 0:
            r_min = round(r_opt + 0.01, 2)
        else:
            r_min = 0.3
        r_max = 1.0
        args.n = str(int(args.n) - 1)
        args.resolution = r_min
        minimal_check = True
        maximal_check = False

        if round(r_max - r_min, 2) <= 0.02 or args.n == "3":
            args.opt = 0
            F_opt = 0
            for key, value in solution.items():
                if "F-score" in value and value["F-score"] > F_opt:
                    args.resolution = round(value["resolution"], 2)
                    args.n = key
                    F_opt = value["F-score"]
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    else:
        if maximal_check:
            args.resolution = round(r_max, 2)
        else:
            args.resolution = round((r_min + r_max) / 2, 2)

"""
opt end at 27
number of views = 4, deadline = 5.0
+ SfM time = [2.94, 2.12, 2.76, 2.77, 2.7, 2.75, 2.61, 2.49, 2.84, 2.77, 2.82, 2.68, 2.6, 2.76, 2.72, 2.78, 2.58, 2.65, 2.56, 2.65, 2.32, 2.86, 2.72, 2.85, 2.86, 2.39, 2.8, 2.23, 2.89, 3.02, 2.93, 1.85, 2.63, 2.42, 2.52, 2.21, 2.04, 2.15, 2.28, 2.08, 2.08, 2.21, 2.45, 1.85, 2.03, 1.91, 2.18, 2.25, 2.23, 1.8, 2.23, 2.16, 2.11, 2.35, 1.92, 2.09, 2.13, 2.53, 2.3, 1.99, 2.13, 2.25, 2.84, 2.81, 2.41, 2.58, 2.74, 2.49, 2.94, 2.12, 2.62, 2.51, 3.06, 2.86, 2.39, 2.78, 2.67, 2.57, 2.94, 2.37, 2.62, 2.37, 2.68, 2.23, 2.5, 2.95, 2.5, 2.58, 2.72, 1.96, 2.8, 2.61, 2.82, 2.46, 1.9, 2.48, 2.51]
+ MVS time = [18.4014, 1.8001, 7.789, 3.7069, 2.3714, 3.3332, 2.6243, 2.9246, 2.6481, 7.6304, 4.7125, 3.401, 2.8252, 3.0227, 3.2465, 2.5188, 5.3196, 3.8777, 3.2266, 3.4012, 3.0413, 2.2389, 3.3528, 2.732, 3.1652, 3.0575, 3.0528, 3.1682, 2.8729, 2.9602, 2.7756, 2.6721, 2.5836, 2.5452, 2.5371, 2.1823, 2.3351, 2.1386, 2.1714, 2.0082, 2.1305, 2.0669, 2.2278, 2.2917, 2.2222, 2.1199, 2.1164, 2.1635, 1.8645, 1.9356, 2.2132, 2.4734, 2.1668, 2.2771, 2.2859, 2.3054, 2.1102, 2.2204, 2.0581, 2.3136, 2.1894, 2.1906, 2.3351, 2.3301, 2.1655, 2.2068, 2.0439, 1.921, 2.148, 1.9108, 2.0658, 1.8719, 2.0221, 1.85, 1.9847, 1.9528, 1.8769, 1.8019, 1.772, 1.8813, 1.8002, 1.7613, 1.8882, 1.874, 1.7297, 1.9622, 1.9033, 1.9679, 1.803, 1.9247, 1.8996, 1.8842, 2.0003, 1.7322, 1.8688, 1.8933, 1.7141]
+ total time = [21.6614, 4.0116, 10.8662, 6.6436, 5.1948, 6.2501, 5.3727, 5.5571, 5.6227, 10.6805, 7.7073, 6.2535, 5.5606, 5.9258, 6.1151, 5.4434, 8.0787, 6.6799, 5.9468, 6.2027, 5.5098, 5.2735, 6.2616, 5.7098, 6.1581, 5.6087, 6.0, 5.5273, 5.9076, 6.1054, 5.8353, 4.6509, 5.3627, 5.0893, 5.1862, 4.4944, 4.4792, 4.4124, 4.5569, 4.1905, 4.3157, 4.4024, 4.7819, 4.3355, 4.3856, 4.1622, 4.4467, 4.5572, 4.2061, 3.8646, 4.5945, 4.7886, 4.4048, 4.7693, 4.3599, 4.571, 4.3757, 4.9004, 4.5064, 4.413, 4.4529, 4.5505, 5.2833, 5.308, 4.7331, 4.9124, 4.9466, 4.5309, 5.2133, 4.154, 4.8225, 4.4871, 5.1906, 4.8102, 4.4762, 4.8568, 4.6496, 4.4731, 4.8023, 4.3426, 4.5097, 4.2425, 4.6582, 4.1941, 4.3257, 5.0228, 4.497, 4.658, 4.6438, 4.0104, 4.7986, 4.5954, 4.9192, 4.323, 3.8902, 4.497, 4.3309]
+ avg total time = 5.281939175257732
+ res = [1.0, 0.3, 1.0, 0.65, 0.47, 0.56, 0.52, 0.54, 0.55, 1.0, 0.78, 0.67, 0.61, 0.64, 0.66, 0.65, 1.0, 0.82, 0.73, 0.77, 0.75, 0.76, 1.0, 0.88, 0.94, 0.91, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.83, 0.81, 0.79, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.75, 0.73, 0.73, 0.73, 0.73, 0.73, 0.71, 0.71, 0.71, 0.71, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67]
+ avg res = 0.7423711340206186
+ F-score = [1.0, 0.1625, 1.0, 0.6354, 0.5024, 0.5526, 0.5298, 0.4995, 0.4226, 0.8279, 0.6379, 0.5613, 0.4763, 0.5394, 0.5249, 0.482, 0.7993, 0.6299, 0.5843, 0.6075, 0.5498, 0.4634, 0.5844, 0.5208, 0.5423, 0.5656, 0.5715, 0.5898, 0.5732, 0.6099, 0.6125, 0.5535, 0.6086, 0.5727, 0.5954, 0.5572, 0.59, 0.537, 0.5066, 0.4968, 0.5388, 0.5272, 0.4978, 0.5463, 0.5062, 0.5038, 0.5349, 0.5229, 0.5183, 0.5221, 0.5067, 0.5102, 0.5551, 0.5418, 0.5742, 0.5589, 0.5642, 0.549, 0.522, 0.4843, 0.4869, 0.4566, 0.4534, 0.4654, 0.4266, 0.4751, 0.4894, 0.4602, 0.479, 0.4184, 0.4179, 0.3942, 0.3794, 0.3643, 0.3896, 0.3638, 0.3953, 0.3886, 0.3883, 0.3529, 0.4143, 0.3839, 0.3613, 0.4201, 0.4219, 0.395, 0.4385, 0.442, 0.4843, 0.4182, 0.466, 0.4495, 0.4469, 0.429, 0.4185, 0.3969, 0.3902]

number of views = 7, deadline = 10.0
+ SfM time = [2.52, 2.12, 2.69, 2.89, 2.74, 2.94, 3.03, 2.52, 7.0, 2.95, 2.77, 2.83, 2.56, 2.59, 2.69, 2.71, 2.52, 2.36, 2.76, 2.68, 2.51, 2.39, 2.58, 2.52, 2.72, 2.14, 2.05, 2.45, 2.46, 2.67, 2.57, 2.27, 2.12, 2.36, 2.37, 2.54, 2.39, 2.56, 2.87, 2.31, 2.64, 2.78, 2.76, 2.71, 2.75, 2.84, 2.72, 2.68, 2.41, 2.92, 2.75, 2.82, 2.82, 2.34, 2.63, 2.11, 2.1, 2.5, 2.49, 2.62, 2.5, 2.46, 2.45, 2.47, 2.93, 2.48, 2.54, 2.65, 2.65, 2.5, 2.75, 2.5, 3.0, 2.67, 2.71, 2.84, 2.52, 2.43, 2.52, 2.66, 2.38, 2.71, 2.77, 2.83, 2.94, 2.76, 2.7, 2.67, 2.46, 2.51, 2.5, 2.54, 2.51, 2.54, 2.26, 2.83, 2.34]
+ MVS time = [18.8343, 1.7837, 7.6803, 8.1063, 8.5548, 8.4704, 8.091, 7.7856, 6.9485, 7.1468, 6.9529, 6.682, 6.6855, 6.603, 6.699, 6.325, 6.4561, 6.6312, 6.7863, 6.6002, 6.4019, 6.3041, 6.1533, 6.343, 6.2052, 6.2677, 6.1822, 6.5624, 6.6351, 6.9479, 6.8853, 6.6162, 6.8739, 7.0947, 6.9345, 7.2888, 6.914, 6.395, 7.082, 6.7165, 6.7876, 7.2183, 7.1031, 7.5937, 7.7497, 7.5465, 7.286, 6.623, 6.3416, 7.2194, 7.9603, 8.3283, 7.8743, 7.6732, 7.4824, 7.2347, 7.1198, 6.9674, 6.9387, 7.1153, 6.9773, 7.1181, 7.0949, 7.3367, 6.8152, 6.9305, 7.114, 7.5177, 7.6439, 7.0763, 7.0007, 6.7689, 7.0733, 7.0852, 7.484, 7.3739, 7.1791, 7.3304, 7.007, 6.5458, 6.8912, 6.732, 6.8721, 8.1595, 8.1671, 7.8889, 7.5527, 7.4976, 7.4175, 6.9883, 6.9856, 6.8177, 6.9276, 6.3734, 6.5936, 6.4714, 6.9136]
+ total time = [21.6519, 3.9932, 10.685, 11.2577, 11.5532, 11.6599, 11.3432, 10.5624, 14.1838, 10.3002, 9.9616, 9.7125, 9.4562, 9.3935, 9.6177, 9.2413, 9.2048, 9.1832, 9.7651, 9.4698, 9.142, 8.8923, 8.9517, 9.048, 9.1816, 8.6016, 8.4391, 9.1986, 9.2921, 9.7969, 9.6602, 9.0595, 9.2156, 9.653, 9.535, 10.0004, 9.5125, 9.1408, 10.1665, 9.2207, 9.6545, 10.2327, 10.0771, 10.5305, 10.7276, 10.6128, 10.2606, 9.5049, 8.9817, 10.3458, 10.9498, 11.378, 10.8858, 10.2435, 10.3407, 9.5763, 9.4175, 9.6783, 9.6114, 9.9852, 9.6816, 9.7839, 9.7307, 10.0137, 9.954, 9.5962, 9.8803, 10.3578, 10.5334, 9.7737, 9.9622, 9.4603, 10.2931, 9.993, 10.389, 10.4402, 9.8889, 9.9879, 9.7044, 9.4106, 9.4697, 9.6545, 9.8661, 11.21, 11.3396, 10.8484, 10.4712, 10.3935, 10.106, 9.6973, 9.6569, 9.5575, 9.6126, 9.1121, 9.0285, 9.497, 9.4488]
+ avg total time = 9.986635051546392
+ res = [1.0, 0.3, 1.0, 1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.88, 0.88, 0.9, 0.92, 0.92, 0.94, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.94, 0.96, 0.98, 0.98, 0.98, 0.96, 0.94, 0.92, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.88, 0.88, 0.9, 0.92, 0.92, 0.9, 0.9, 0.9, 0.92, 0.92, 0.94, 0.92, 0.9, 0.9, 0.9, 0.9, 0.92, 0.94, 0.96, 0.98, 0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.9, 0.92]
+ avg res = 0.9014432989690722
+ F-score = [1.0, 0.1625, 1.0, 1.0, 0.8795, 0.8399, 0.8248, 0.843, 0.8068, 0.8074, 0.7817, 0.7765, 0.7814, 0.7795, 0.8039, 0.7991, 0.8011, 0.7998, 0.8176, 0.7994, 0.7939, 0.7982, 0.8139, 0.7843, 0.8185, 0.8203, 0.8017, 0.8346, 0.8178, 0.8071, 0.8018, 0.8216, 0.8069, 0.7958, 0.8104, 0.8376, 0.8498, 0.8528, 0.8478, 0.8326, 0.8571, 0.8547, 0.8636, 0.8393, 0.8729, 0.8625, 0.8552, 0.8677, 0.8711, 0.8672, 0.854, 0.8701, 0.8499, 0.8521, 0.8639, 0.8322, 0.8329, 0.8344, 0.8267, 0.8133, 0.8175, 0.8227, 0.8135, 0.828, 0.8357, 0.848, 0.8311, 0.8543, 0.8533, 0.825, 0.8431, 0.8416, 0.8483, 0.8647, 0.8578, 0.8519, 0.8391, 0.8325, 0.8359, 0.8419, 0.8507, 0.8477, 0.8513, 0.8889, 0.8717, 0.8512, 0.8536, 0.8463, 0.826, 0.8309, 0.8464, 0.8429, 0.8533, 0.8555, 0.8547, 0.8548, 0.8546]

opt end at 12
number of views = 7, deadline = 7.5
+ SfM time = [2.7, 2.04, 2.69, 2.57, 2.66, 2.66, 2.8, 2.25, 2.57, 2.91, 2.88, 2.52, 2.83, 2.57, 2.73, 2.73, 2.83, 2.28, 2.06, 2.34, 2.12, 1.91, 2.38, 2.38, 2.26, 1.91, 2.08, 2.3, 2.37, 2.41, 2.17, 1.7, 2.19, 1.73, 2.35, 2.45, 1.92, 1.69, 1.68, 1.7, 1.65, 1.79, 1.58, 1.7, 1.95, 1.71, 1.8, 1.83, 2.06, 1.94, 2.03, 2.01, 1.96, 2.0, 2.04, 2.11, 2.05, 2.18, 1.95, 2.28, 1.97, 1.96, 2.53, 2.05, 2.39, 2.43, 2.07, 2.17, 2.49, 1.82, 2.29, 2.16, 1.97, 2.33, 2.42, 2.69, 2.54, 1.9, 2.27, 2.43, 2.24, 2.1, 2.07, 2.19, 2.03, 2.24, 2.15, 2.11, 2.24, 1.84, 2.14, 2.29, 2.34, 2.05, 2.11, 1.96, 2.01]
+ MVS time = [18.6104, 1.7729, 8.0792, 3.6824, 6.1323, 5.1185, 5.5821, 5.3861, 4.6032, 7.7796, 6.116, 3.3732, 5.5821, 4.4037, 4.9852, 4.9935, 5.5459, 5.178, 5.0025, 4.8808, 5.1752, 4.5415, 4.4916, 4.3121, 4.4993, 4.5333, 4.4864, 4.7083, 4.6936, 4.9959, 4.5388, 4.6824, 4.5954, 4.771, 4.3343, 4.3685, 4.3123, 4.0358, 4.2147, 4.0567, 3.8745, 4.0626, 3.964, 4.0843, 4.016, 4.2848, 4.3962, 4.6245, 4.5692, 5.0024, 5.7168, 5.8933, 5.588, 6.0243, 5.9619, 6.1303, 5.7496, 5.854, 5.3396, 5.3844, 5.2505, 5.4951, 5.08, 5.0427, 4.9067, 4.7529, 4.7501, 4.8919, 5.5289, 5.1001, 5.1274, 4.9181, 5.3963, 5.1576, 4.7948, 5.2187, 5.0425, 4.8016, 4.9031, 4.7201, 5.0476, 5.0259, 5.2849, 5.7281, 5.6235, 4.9985, 5.3734, 5.0706, 5.2176, 4.9828, 4.756, 5.2294, 5.0184, 4.8755, 4.8968, 5.417, 5.4143]
+ total time = [21.6129, 3.8983, 11.0608, 6.4198, 9.026, 7.9604, 8.6002, 7.81, 7.3344, 10.9358, 9.1978, 6.0637, 8.6095, 7.1661, 7.9357, 7.8982, 8.5812, 7.6253, 7.2703, 7.414, 7.513, 6.6481, 7.0468, 6.8725, 6.9256, 6.6191, 6.7506, 7.167, 7.2175, 7.5866, 6.8662, 6.5556, 6.9439, 6.6422, 6.8424, 6.9489, 6.3657, 5.8797, 6.053, 5.9095, 5.7049, 6.0088, 5.6924, 5.9657, 6.1145, 6.1735, 6.347, 6.6128, 6.8233, 7.1105, 7.9389, 8.1079, 7.8053, 8.1916, 8.1698, 8.4342, 7.9903, 8.2311, 7.4609, 7.8668, 7.3832, 7.6594, 7.7761, 7.2594, 7.4704, 7.3348, 7.018, 7.2274, 8.1832, 7.1012, 7.5925, 7.2598, 7.5443, 7.6774, 7.3706, 8.0703, 7.7585, 6.8703, 7.3562, 7.3022, 7.4444, 7.333, 7.5372, 8.1302, 7.8214, 7.4459, 7.7084, 7.3787, 7.6159, 6.9662, 7.0688, 7.6715, 7.5296, 7.0836, 7.1857, 7.56, 7.5964]
+ avg total time = 7.492954639175258
+ res = [1.0, 0.3, 1.0, 0.65, 0.82, 0.73, 0.77, 0.75, 0.76, 1.0, 0.88, 0.77, 1.0, 0.89, 0.95, 0.97, 0.98, 0.75, 0.73, 0.73, 0.73, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.79, 0.79, 0.77, 0.77, 0.75, 0.73, 0.73, 0.73, 0.73, 0.75, 0.77, 0.75, 0.77, 0.75, 0.77, 0.77, 0.75, 0.77, 0.75, 0.73, 0.73, 0.75, 0.77, 0.79, 0.81, 0.81, 0.79, 0.77, 0.77, 0.75, 0.75, 0.73, 0.73, 0.75, 0.75, 0.75, 0.77, 0.79, 0.79]
+ avg res = 0.7608247422680413
+ F-score = [1.0, 0.1625, 1.0, 0.6764, 0.8246, 0.7045, 0.7506, 0.7605, 0.6348, 0.8279, 0.6955, 0.5537, 0.7812, 0.6257, 0.6563, 0.6961, 0.7026, 0.7425, 0.7228, 0.7416, 0.7302, 0.6897, 0.701, 0.6886, 0.7296, 0.7269, 0.7222, 0.744, 0.7316, 0.7376, 0.737, 0.7277, 0.7202, 0.7181, 0.7188, 0.7364, 0.7459, 0.7465, 0.7217, 0.6954, 0.7139, 0.6956, 0.7344, 0.6666, 0.6517, 0.698, 0.7381, 0.7859, 0.7822, 0.7982, 0.7615, 0.7937, 0.8015, 0.7827, 0.7927, 0.7771, 0.7935, 0.789, 0.782, 0.7633, 0.769, 0.7777, 0.7642, 0.7437, 0.7572, 0.7598, 0.7469, 0.764, 0.7824, 0.7754, 0.7996, 0.7711, 0.8011, 0.7975, 0.7769, 0.7852, 0.7918, 0.7481, 0.7601, 0.7654, 0.8101, 0.8019, 0.7882, 0.7911, 0.8057, 0.8141, 0.7923, 0.8002, 0.7909, 0.7576, 0.7622, 0.7846, 0.7795, 0.7933, 0.814, 0.7997, 0.7967]

golden
+ SfM time = [3.43, 2.91, 2.94, 2.86, 2.9, 2.89, 2.95, 3.02, 2.95, 2.92, 2.93, 2.98, 2.91, 2.86, 2.93, 2.95, 2.81, 2.95, 2.92, 3.06, 2.89, 2.9, 2.89, 2.93, 2.85, 2.89, 2.88, 3.08, 2.86, 2.88, 2.99, 3.0, 2.89, 2.88, 2.89, 3.0, 2.91, 2.98, 2.9, 3.12, 2.88, 3.01, 2.78, 2.99, 3.07, 2.95, 2.98, 2.86]
+ MVS time = [17.6817, 7.847, 8.202, 8.927, 9.0626, 9.2002, 9.3481, 9.2432, 9.1537, 9.0508, 9.2454, 8.7647, 8.8384, 8.5609, 8.7069, 8.5124, 8.5472, 8.8791, 8.8501, 8.8386, 8.5566, 8.4933, 8.3392, 8.273, 8.0525, 8.2822, 8.2988, 8.8032, 8.6832, 9.2776, 9.0453, 8.9651, 9.1463, 9.1418, 8.9891, 9.1544, 8.7872, 7.9729, 8.1369, 8.0171, 7.717, 8.0141, 7.6895, 8.197, 8.3753, 8.2294, 7.9138, 7.5108]
+ total time = [21.6345, 11.0511, 11.4053, 12.0886, 12.2912, 12.3799, 12.5445, 12.5257, 12.3904, 12.2384, 12.4437, 12.0051, 11.9901, 11.6743, 11.8907, 11.7021, 11.6275, 12.1114, 11.9988, 12.1464, 11.6943, 11.6352, 11.4775, 11.4594, 11.13, 11.4229, 11.4321, 12.1113, 11.8125, 12.4125, 12.262, 12.2089, 12.2898, 12.2297, 12.1102, 12.3881, 11.9279, 11.2459, 11.286, 11.3755, 10.87, 11.2855, 10.6934, 11.4338, 11.6956, 11.4082, 11.1463, 10.6263]
+ avg total time = 11.983552083333334
+ res = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
+ avg res = 1.0
+ F-score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

+ SfM time = [2.79, 2.5, 2.54, 2.5, 2.58, 2.68, 3.2, 2.81, 2.78, 2.73, 2.81, 3.01, 3.16, 2.83, 2.83, 2.73, 3.33, 3.23, 2.59, 3.46, 3.15, 3.12, 3.2, 3.05, 3.12, 3.16, 3.27, 3.12, 3.2, 3.16, 3.18, 3.16, 3.16, 3.15, 3.02, 2.99, 3.06, 2.96, 3.07, 3.2, 3.18, 3.02, 3.22, 3.07, 3.18, 3.06, 3.11, 3.03]
+ MVS time = [13.4759, 4.8854, 5.0463, 4.8716, 5.1834, 5.1421, 5.1877, 5.5189, 5.4818, 5.7828, 5.6367, 5.8412, 5.8747, 6.0814, 7.1477, 5.9128, 5.8411, 5.3475, 5.4508, 5.1523, 5.2837, 5.3843, 5.519, 5.4733, 5.6932, 5.7406, 5.5763, 5.8057, 5.9507, 5.7434, 5.9497, 5.4002, 5.376, 5.2961, 5.2695, 5.4982, 5.2266, 5.7493, 5.6943, 5.4617, 5.401, 5.2037, 5.4247, 5.4137, 5.4492, 5.3045, 5.2775, 5.1407]
+ total time = [16.5738, 7.6117, 7.8308, 7.6147, 7.9717, 8.0641, 8.6357, 8.561, 8.4767, 8.6942, 8.6749, 9.0286, 9.238, 9.0907, 10.1822, 8.8354, 9.4297, 8.7564, 8.2655, 8.7958, 8.6625, 8.6994, 8.9397, 8.7137, 9.0813, 9.0905, 9.0481, 9.1086, 9.4655, 9.0783, 9.3336, 8.7396, 8.7662, 8.6348, 8.4934, 8.694, 8.472, 8.9217, 8.9472, 8.8664, 8.751, 8.4244, 8.8417, 8.699, 8.8054, 8.5712, 8.572, 8.3957]
+ avg total time = 8.898927083333334
+ res = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
+ avg res = 1.0
+ F-score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
"""