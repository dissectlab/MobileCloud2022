import json
import copy
'''
    1. From the result (.json) of OpenMVG of original image set
       Extract:
        - background 3D points (x,y,z locations)
        - corresponding 2D points in each image

    2. Write a new .json file for background               
'''


# F             -> original .json file
# BGImagePath   -> path to background images
# msa           -> mask size array
#                  if n points, then 4*n elements in array
def writeBG_FG(F, BGImagePath, FGImagePath, msa, savePath):
    f = open(F, 'r')
    F_bg = json.load(f)
    F_fg = copy.deepcopy(F_bg)

    F_bg["root_path"] = BGImagePath  # rewrite the path to BGImagePath
    F_fg["root_path"] = FGImagePath  # rewrite the path to BGImagePath

    struc = F_bg['structure']
    F_fg["structure"] = []

    i = 0
    k = 0

    L = len(struc)

    #####################
    #   Simple search   #
    #####################

    while i < len(struc):
        num_obsevations = len(struc[i]['value']['observations'])
        for j in range(num_obsevations):

            if struc[i]['value']['observations'][j]['key'] == 0:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[0] and x <= msa[1]) and (y >= msa[2] and y <= msa[3]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 1:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[4] and x <= msa[5]) and (y >= msa[6] and y <= msa[7]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 2:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[8] and x <= msa[9]) and (y >= msa[10] and y <= msa[11]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 3:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[12] and x <= msa[13]) and (y >= msa[14] and y <= msa[15]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 4:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[16] and x <= msa[17]) and (y >= msa[18] and y <= msa[19]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 5:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[20] and x <= msa[21]) and (y >= msa[22] and y <= msa[23]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

            if struc[i]['value']['observations'][j]['key'] == 6:
                x = struc[i]['value']['observations'][j]['value']['x'][0]
                y = struc[i]['value']['observations'][j]['value']['x'][1]
                if (x >= msa[24] and x <= msa[25]) and (y >= msa[26] and y <= msa[27]):
                    F_fg["structure"].append(struc[i])
                    del struc[i]
                    i = i - 1
                    break
                continue

        i = i + 1
    #print("Total points:", L)
    #print("After, BG 3D points:", len(F_bg["structure"]))
    #print("After, FG 3D points:", len(F_fg["structure"]))

    foo = open(savePath + "/bg.json", "w")
    foo.write(
        json.dumps(F_bg, sort_keys=False, indent=4, separators=(',', ':'))
    )
    f.close()
    foo.close()

    foo = open(savePath + "/fg.json", "w")
    foo.write(
        json.dumps(F_fg, sort_keys=False, indent=4, separators=(',', ':'))
    )
    f.close()
    foo.close()

    return L, len(F_bg["structure"]), len(F_fg["structure"])

