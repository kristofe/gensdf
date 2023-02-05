# libigl --> HAS TO BE IN A DIFFERENT VIRTUAL ENVIRONMENT THAN PYTORCH
# conda install -c conda-forge igl 
# conda install -c conda-forge meshplot 
#
# https://libigl.github.io/libigl-python-bindings/tutorials/
#
# for better meshes and normal/gradient calculation use the 
# following to convert shapenet
# https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion

import igl
import h5py
import os
import argparse 
import pathlib
cwd = os.getcwd()


def playground():
    print(cwd)

    filename = f"{cwd}/data/acronym/grasps/1Shelves_1e3df0ab57e8ca8587f357007f9e75d1_0.011099225885734912.h5"

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        #ds_arr = f[a_group_key][()]  # returns as a numpy array

        model = f['object/file'][()]
        filepath = pathlib.Path(str(model)) 
        shapenet_id = filepath.stem
        object_class = filepath.parts[-2:-1]
        print(filepath.parts)

def load_mesh(path):
    return igl.read_triangle_mesh(str(path))


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument('--split_file', '-s', default='sv2_sofas_train.json')
    #arg_parser.add_argument('--root_dir', '-r', default='../DeepSDF/data/SdfSamples/acronym')
    arg_parser.add_argument('--shapenetsem_root_dir', '-r', default='~/Documents/Projects/datasets/shapenet_sem/models/')
    arg_parser.add_argument('--acronym_root_dir', '-r', default='../acronym/grasps/')
    arg_parser.add_argument('--glob_pattern', '-r', default='*.h5')
    arg_parser.add_argument('--class_name', '-c', nargs="+")
    #arg_parser.add_argument('--output_dir', '-o', default='../DeepSDF/data/SDFSamples/ShapeNetV2')
    #arg_parser.add_argument('--mkdir', action='store_true')

    args = arg_parser.parse_args()

    root = pathlib.Path(args.acronym_root_dir)
    shapenetsem_root = pathlib.Path(args.shapenetsem_root_dir)
    all_paths = root.glob(args.glob_pattern)
    paths = []
    for h5_filename in all_paths:
        if h5_filename.is_file:
            with h5py.File(h5_filename, "r") as f:
                model = f['object/file'][()]
                filepath = pathlib.Path(str(model)) 
                shapenet_id = filepath.stem
                object_class = filepath.parts[-2:-1]
                shapenet_model_path = f'{args.shapenetsem_root_dir}{str(shapenet_id)}.obj'
                paths.append(shapenet_model_path)
    
    model_count = len(paths)
    model_counter = 1
    for model_path in paths:
        print(model_path)
        v, f = load_mesh(model_path)
        print(f"loaded {model_counter}/{model_count}")
        model_counter += 1
        if v.shape[0] == 0 or f.shape[0] == 0:
            continue

        break
                