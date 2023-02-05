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
import numpy as np
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

def importance_rejection(self, sdf, beta, max_samples, split=0.9):
    score_channel = 3
    score = np.exp((-beta)*np.abs(sdf[:,score_channel])) 
    noise = np.random.randn(score.shape[0]) * 0.1
    score = score + noise

    score_indices = np.argsort(score)[::-1] #Reverse so sort is ascending 
    sdf = sdf[score_indices]

    #This is now sorted.  
    if split < 1.0 and split >= 0.0:
        # take highest importance scored subset of the samples leaving room for random points
        max_importance = int(split*max_samples)  # get index of last sample
        max_random = max_samples - max_importance # calculate set size of random samples
        important_samples = sdf[:max_importance] # get ranked subset

        #Now choose randomly from all of the samples that were not part of the importance sampling.
        # just choose a bunch of random indices whose length is the set size of the random samples
        random_indices = np.random.randint(low=max_importance, high=sdf.shape[0], size=max_random)
        random_unimportant_samples = sdf[random_indices] # grab the random samples

        sdf = np.concatenate([important_samples,random_unimportant_samples], axis=0)
    else:
        sdf = sdf[:max_samples]
    return sdf


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument('--split_file', '-s', default='sv2_sofas_train.json')
    #arg_parser.add_argument('--root_dir', '-r', default='../DeepSDF/data/SdfSamples/acronym')
    arg_parser.add_argument('--shapenetsem_root_dir', default='~/Documents/Projects/datasets/shapenet_sem/models/')
    arg_parser.add_argument('--acronym_root_dir', default='../acronym/grasps/')
    arg_parser.add_argument('--glob_pattern',  default='*.h5')
    arg_parser.add_argument('--class_name', '-c', nargs="+")
    arg_parser.add_argument('--output_dir', '-o', default='~/Documents/Projects/datasets/acronym_sdf_samples/')
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
                paths.append((shapenet_model_path, shapenet_id, object_class))
    
    model_count = len(paths)
    model_counter = 1
    sample_count = 1000000
    target_sample_count = 250000
    importance_beta = 30
    for (model_path, object_id, class_id) in paths:
        print(model_path)
        if os.path.exists(model_path):
            print("Correct Path")
        
        #v, f = load_mesh(model_path)
        print(f"loaded {model_counter}/{model_count}")
        model_counter += 1
        if v.shape[0] == 0 or f.shape[0] == 0:
            continue


        #FIXME: Sample points logic should be lifted from GenSDF logic (c++ code)
        #This will take the sample points outside the -0.5 to 0.5 range!!!... Center the model before normalizing!!!!!
        sample_count = int(sample_count)
        points = np.random.uniform(-0.5, 0.5, (sample_count,3))
        # Thingy10k has 22% non-manifold models so the pseudonormal method can't be used
        # ACRONYM should have 100% manifold meshes so pseudonormal method (the fast one) can be used
        # The following function is multithreaded and will use all of the cpu cores on a machine.
        (signed_distances, face_indices, closest_points) = igl.signed_distance(points,v, f, return_normals=True)

        sdf = np.stack((points[:,0], points[:,1], points[:,2], signed_distances), axis=-1)
        sdf = importance_rejection(sdf, beta=importance_beta, max_samples=target_sample_count, split=1.0)
        valid_sample_count = sdf.shape[0]
        np.random.shuffle(sdf)
        target_path = args.output_dir
        np.savez(target_path,sdf_points=sdf.astype(np.float32), filename=model_path, beta=importance_beta, classid=class_id, modelid=object_id)
        print(f'saved {target_path}')

        break
                