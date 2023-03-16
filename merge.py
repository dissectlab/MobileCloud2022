import numpy as np
from plyfile import PlyData, PlyElement


def merge(fg, bg, result):
    fg_ply = PlyData.read(fg)
    bg_ply = PlyData.read(bg)

    fg_vertex = fg_ply.elements[0].data
    bg_vertex = bg_ply.elements[0].data

    with open(fg, 'rb') as f:
        fg_rows = f.readlines()

    with open(bg, 'rb') as f:
        bg_rows = f.readlines()[13:]

    fg_rows[2] = ('element vertex ' + str(len(fg_vertex) + len(bg_vertex)) + '\n').encode()
    fg_rows = np.concatenate([fg_rows, bg_rows]).tolist()

    f = open(result, "wb")
    for item in fg_rows:
        f.write(item)
    f.close()


if __name__ == '__main__':
    # bg_path = "/home/zxj/zxj/3d-reconstruction/data/gold_bg/00001_scene_dense_mesh_refine_texture.ply"
    for i in range(1, 48):
        try:
            merge("/home/zxj/zxj/3d-reconstruction/data/all/" + str(i).zfill(
                5) + "_output/fg_mvs/scene_dense_mesh_refine_texture.ply",
                  "/home/zxj/zxj/3d-reconstruction/data/gold_bg_2/" + str(2).zfill(
                      5) + "_scene_dense_mesh_refine_texture.ply",
                  "/home/zxj/zxj/3d-reconstruction/data/results_2/result_" + str(i).zfill(5) + '.ply')
        except Exception as e:
            print(e)
