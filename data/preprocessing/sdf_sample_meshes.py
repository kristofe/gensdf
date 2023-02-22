# pip install libigl, h5py
# https://github.com/libigl/libigl-python-bindings/blob/main/tutorial/igl_docs.md
#
# for better meshes and normal/gradient calculation use the 
# following to convert shapenet
# https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion

'''
Using the full ACRONYM dataset
Download the full dataset (1.6GB): acronym.tar.gz
Download the ShapeNetSem meshes from https://www.shapenet.org/

Create watertight versions of the downloaded meshes:
Clone and build: https://github.com/hjwdzh/Manifold

(There is a better version: https://github.com/hjwdzh/ManifoldPlus but clearly the version below worked for the paper)
Create a watertight mesh version assuming the object path is model.obj:
    manifold model.obj temp.watertight.obj -s
Simplify it: 
    simplify -i temp.watertight.obj -o model.obj -m -r 0.02

For more details about the structure of the ACRONYM dataset see: https://sites.google.com/nvidia.com/graspdataset


####
Installing manifold

Install into root of project/manifold

git clone --recursive -j8 git@github.com:hjwdzh/Manifold.git manifold
cd manifold
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

'''
import igl
import h5py
import numpy as np
import os
import argparse 
import pathlib
import trimesh
import torch
import math
import subprocess
import shutil

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


def gensdf_sample(path, object_id, class_id, target_path):

    print(f"Reading the input mesh {path}")
    m = trimesh.load(path)
    #igl_v, igl_f = igl.read_triangle_mesh(str(path))


    if(m.is_watertight == False):
        print(f"{path} is not watertight")
        #raise(Exception(f"{path} is not watertight"))
    '''
    #because the meshes are processed they aren't scenes but are trimeshes
    scene = trimesh.load(path)
    meshes = []
    for key, geom in scene.geometry.items():
        meshes.append(geom)
        if(geom.is_watertight == False):
            print(f"{path} is not watertight")
            break
            #raise(Exception(f"{path} is not watertight"))
    m = trimesh.util.concatenate(meshes)
    '''

    #recenter mesh
    print("recentering")
    centroid = m.centroid
    v = m.vertices
    f = m.faces
    v = v - centroid

    #normalize bbox
    print("normalizing mesh")
    max = np.amax(v,axis=0)
    min = np.amin(v,axis=0)
    norm = np.linalg.norm(max - min)
    v = v/norm
    
    #m.update_vertices(v)


    print("sampling points")
    num_samples = 500000
    num_samples_from_surface = (int)(47 * num_samples / 100)
    num_samples_near_surface = num_samples - num_samples_from_surface
    pc = trimesh.sample.sample_surface(m, num_samples_from_surface)
    pc = torch.from_numpy(pc[0])

    pc = pc - centroid
    pc = pc/norm

    print("sampling query points...")
    variance = 0.005
    second_variance = variance / 10.0
    perturb_norm1 = torch.normal(mean=0.0, std=math.sqrt(variance), size=(num_samples_from_surface,3) )
    perturb_norm2 = torch.normal(mean=0.0, std=math.sqrt(second_variance),size=(num_samples_from_surface,3))
    #REPLACE BY PYTORCH or NUMPY normal

    querypoints1 = pc + pc * perturb_norm1
    querypoints2 = pc + pc * perturb_norm2

    querypoints = torch.cat((pc, querypoints1,querypoints2), dim=0)
    #querypoints = torch.cat([pc,pc], dim=0)

    print("computing signed distances...")
    (signed_distances, face_indices, closest_points, normals) = igl.signed_distance(querypoints.numpy(),v, f, return_normals=True)

    #zero_ids = np.abs(signed_distances) < 1e-8
    #signed_distances[zero_ids] = 0.0

    print("saving results...")
    sdf = np.stack((querypoints[:,0], querypoints[:,1], querypoints[:,2], signed_distances), axis=-1)
    #np.random.shuffle(sdf)

    np.savez(target_path,sdf_points=sdf.astype(np.float32), filename=str(path), beta=importance_beta, classid=class_id, modelid=object_id)
    print(f'saved {target_path}')

def importance_rejection(sdf, beta, max_samples, split=0.9):
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
    arg_parser.add_argument('--shapenetsem_root_dir', default='../datasets/shapenet_sem/models/')
    arg_parser.add_argument('--acronym_root_dir', default='data/acronym/grasps/')
    arg_parser.add_argument('--glob_pattern',  default='*.h5')
    arg_parser.add_argument('--class_name', '-c', nargs="+")
    arg_parser.add_argument( "--max_model_count", type=int, default=-1)
    arg_parser.add_argument( "--separate_folders", type=int, default=1)
    arg_parser.add_argument('--output_dir', '-o', default='../datasets/shapenet_sem/processed/')
    arg_parser.add_argument( "--force", type=int, default=0)
    #arg_parser.add_argument('--mkdir', action='store_true')

    args = arg_parser.parse_args()

    force = False if args.force == 0 else True

    root = pathlib.Path(args.acronym_root_dir)
    shapenetsem_root = pathlib.Path(args.shapenetsem_root_dir)
    all_paths = root.glob(args.glob_pattern)
    paths = []

    #args.max_model_count = 8
    counter = 0
    max_model_count = 100000000 if args.max_model_count == -1 else args.max_model_count
    for h5_filename in all_paths:
        if h5_filename.is_file:
            with h5py.File(h5_filename, "r") as f:
                model = f['object/file'][()]
                filepath = pathlib.Path(str(model)) 
                shapenet_id = filepath.stem
                object_class = filepath.parts[-2:-1][0]
                shapenet_model_path = f'{args.shapenetsem_root_dir}{str(shapenet_id)}.obj'
                paths.append((shapenet_model_path, shapenet_id, object_class))
            counter += 1
            if(counter >= max_model_count):
                break
    
    model_count = len(paths)
    model_counter = 1
    sample_count = 2000000
    target_sample_count = 250000
    importance_beta = 30


    for (model_path, object_id, class_id) in paths:
        print(f"loading {model_counter}/{model_count}")
        print(model_path)
        model_counter += 1

        if not os.path.exists(model_path):
            print(f"cannot find {model_path}")


        output_dir = args.output_dir if args.separate_folders == 0 else f"{args.output_dir}/{object_id}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        temp_model_path = f"{model_path[:-4]}_tmp_manifold.obj"
        manifold_model_path = f"{model_path[:-4]}_manifold.obj"
        npz_output_path = f"{output_dir}{object_id}.npz"

        if(not os.path.exists(manifold_model_path) or force):
            #Make sure the model is watertight
            command_line = f"manifold/build/manifold {cwd}/{model_path}  {cwd}/{temp_model_path} -s"
            print(f'{command_line}')
            popen = subprocess.Popen(command_line, stderr=subprocess.DEVNULL, shell=True)
            popen.wait()

            if(not os.path.exists(temp_model_path)):
                print(f"Failed to make {model_path} manifold")
                continue

            #command_line = f"manifold/build/simplify -i {temp_model_path} -o {manifold_model_path} -m -r 0.02; rm {temp_model_path}"
            command_line = f"manifold/build/simplify -i {temp_model_path} -o {manifold_model_path} -m -r 0.02"
            #print(f'{command_line}')
            popen = subprocess.Popen(command_line, stderr=subprocess.DEVNULL, shell=True)
            popen.wait()

            if(not os.path.exists(manifold_model_path)):
                print(f"Couldn't simplify {model_path} using full resolution")
                shutil.copyfile(temp_model_path, manifold_model_path)
                if os.path.isfile(temp_model_path):
                    os.remove(temp_model_path)

        if(os.path.exists(manifold_model_path) and
          (not os.path.exists(npz_output_path) or force)):
            #FIXME: NEEDS TO BE CENTERED AND INSIDE UNIT CUBE: See gensdf sample
            try:
                v, f = load_mesh(manifold_model_path)
            except:
                print(f"Couldn't load {model_path} skipping")
                continue

            try:
                m = trimesh.load(manifold_model_path)
            except:
                print(f"Couldn't load {model_path} skipping")

            #recenter mesh
            print(f"recentering {len(m.vertices)} vertices")
            centroid = m.centroid
            v = m.vertices
            f = m.faces
            v = v - centroid

            #normalize bbox
            print("normalizing mesh")
            max = np.amax(v,axis=0)
            min = np.amin(v,axis=0)
            norm = np.linalg.norm(max - min)
            v = v/norm
            

            if v.shape[0] == 0 or f.shape[0] == 0:
                continue

            #FIXME: Sample points logic should be lifted from GenSDF logic (c++ code)
            #This will take the sample points outside the -0.5 to 0.5 range!!!... Center the model before normalizing!!!!!
            sample_count = int(sample_count)
            points = np.random.uniform(-0.5, 0.5, (sample_count,3))
            # Thingy10k has 22% non-manifold models so the pseudonormal method can't be used
            # ACRONYM should have 100% manifold meshes so pseudonormal method (the fast one) can be used
            # The following function is multithreaded and will use all of the cpu cores on a machine.
            print("computing distances")
            (signed_distances, face_indices, closest_points, normals) = igl.signed_distance(points,v, f, return_normals=True)

            sdf = np.stack((points[:,0], points[:,1], points[:,2], signed_distances), axis=-1)
            sdf = importance_rejection(sdf, beta=importance_beta, max_samples=target_sample_count, split=0.95)
            valid_sample_count = sdf.shape[0]
            np.random.shuffle(sdf)

            
            np.savez(npz_output_path,sdf_points=sdf.astype(np.float32), filename=manifold_model_path, beta=importance_beta, classid=class_id, modelid=object_id)
            print(f'saved {npz_output_path}')

        pc_sampling_target_path = f"{output_dir}{object_id}_gensdf_sampling.npz"
        if(os.path.exists(manifold_model_path) and 
          (not os.path.exists(pc_sampling_target_path) or force)):
            # really inefficient to reopen the mesh etc.  just testing right now.
            print("sampling point cloud")
            gensdf_sample(manifold_model_path, object_id, class_id, pc_sampling_target_path)


        # No need to run c++ sampler. python version does the same thing.
        #command_line = ("data/preprocessing/build/sdf_gen",  manifold_model_path, output_dir, "|| true")
        #print(f'Running {command_line} from cdw {os.getcwd()}')
        #popen = subprocess.Popen(command_line, stderr=subprocess.DEVNULL)
        #popen.wait()

        material_path = f"{model_path[:-4]}.mtl"
        manifold_model_path = f"{model_path[:-4]}_manifold.obj"
        command_line = f"cp {model_path} {material_path} {manifold_model_path} {output_dir}/"
        popen = subprocess.Popen(command_line, stderr=subprocess.DEVNULL, shell=True)
        popen.wait()

                