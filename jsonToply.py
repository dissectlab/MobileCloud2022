import json
from plyfile import PlyData, PlyElement


def py(input_path, output_path):
    #############################################################
    # To get 3D points' location from .json file
    #############################################################
    F = json.load(open(input_path))
    # print(F)

    struc = F['structure']
    n = len(struc)
    #print(n)

    position_3D = []

    for i in range(n):
        position_3D.append(struc[i]['value']['X'])

    #############################################################
    # To write .ply file
    #############################################################
    with open(output_path, 'w') as f:
        f.seek(0)
        f.write('ply\nformat ascii 1.0\ncomment VCGLIB generated\n')
        f.write('element vertex ')
        f.write(str(n))
        f.write('\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')

        for i in range(n):
            for j in range(3):
                f.write(str(position_3D[i][j]))
                f.write(' ')
            f.write('\n')


if __name__ == '__main__':
    point = []
    for timestamp in range(4, 60):
        str_timestamp = str(timestamp).zfill(5)
        path = '/home/ubuntu/3d-reconstruction/data/eva/1.0_bg_once/' + str_timestamp + '_output/sfm/fg.json'
        F = json.load(open(path))
        struc = F['structure']
        point.append(len(struc))

    print(list(point))

