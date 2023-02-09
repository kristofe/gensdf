import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="../datasets/shapenet_sem/processed/", type=str, metavar="DIR")
    parser.add_argument("--sample_count", type=int, default=10000000)
    parser.add_argument("--target_sample_count", type=int, default=1000000)
    parser.add_argument("--importance_beta", type=int, default=30) #Overfit paper was 30
    parser.add_argument("--is_grid", action="store_true", default=False)

    args = parser.parse_args()
    #args.dataset_dir = "../datasets/shapenet_sem/processed/"
    fname = "9ff8c2118a1e9796be58c5ebb087be4f/9ff8c2118a1e9796be58c5ebb087be4f_gensdf_sampling.npz"
    #fname = "9ff8c2118a1e9796be58c5ebb087be4f/9ff8c2118a1e9796be58c5ebb087be4f.npz"
    #fname = "9ff8c2118a1e9796be58c5ebb087be4f/sdf_data.csv"
    if(fname[-4:] == ".csv"):
        df_points =pd.read_csv(args.dataset_dir + fname, sep=',',header=None).values
    else:
        with np.load(args.dataset_dir + fname) as f:
            df_points = f['sdf_points'] 
            filename = f['filename'] 
            importance_beta = f['beta']
    print(f"loaded {fname} shape: {df_points.shape}")
    
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    GRID = False #args.is_grid
    # FOR RANDOM POINT LIST
    if GRID == False:
        count = 50000
        random_indices = np.random.randint(low=0, high=df_points.shape[0], size=count)
        d = df_points[random_indices]
        print(f'min {d.min()} max {d.max()}')
        c = np.size(np.where(d[:,3] < 0.0))
        print(f'neg values {d.shape[0]}/{c}')#np.sum(d[:,3] < 0.0)}')
        '''
        xs = df_points[:count,0]
        ys = df_points[:count,1]
        zs = df_points[:count,2]
        r = df_points[:count,3]
        '''
        sc = ax.scatter(d[:,0],d[:,1],d[:,2],c=d[:,3], marker='.', cmap='Spectral')
    else:
        # FOR GRID
        num_elem = int(df_points.shape[0]**3)
        positions = df_points[:,:,:,0:3]
        values = df_points[:,:,:,3:4]
        print(f'pos min {positions.min()} pos max {positions.max()}')
        print(f'val min {values.min()} max {values.max()}')
        c = np.size(np.where(values[:,:,:,0] < 0.0))
        print(f'neg values {c}/{num_elem}')#np.sum(d[:,3] < 0.0)}')

        sc = ax.scatter(positions[:,:,:,0].view(-1),positions[:,:,:,1].view(-1),positions[:,:,:,2].view(-1),c=values[:,:,:,:].view(-1), marker='.', cmap='Spectral')
    plt.colorbar(sc)
    plt.show()