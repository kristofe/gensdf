import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="shapenet/mesh_sdfs/", type=str, metavar="DIR")
    parser.add_argument("--sample_count", type=int, default=10000000)
    parser.add_argument("--target_sample_count", type=int, default=1000000)
    parser.add_argument("--importance_beta", type=int, default=30) #Overfit paper was 30
    parser.add_argument("--is_grid", action="store_true", default=False)

    args = parser.parse_args()
    args.dataset_dir = "eraseme/"
    fname = "d18f2aeae4146464bd46d022fd7d80aa.npz"
    with np.load(args.dataset_dir + fname) as f:
        df_pointspos = f['neg'] 
        df_pointsneg = f['pos'] 
        df_points = np.concatenate((df_pointsneg,df_pointspos), axis=0)
    '''
    args.dataset_dir = "shapenet/mesh_sdfs_test/"
    fname = "dist100_chair_4918512f624696278b424343280aeccb_points_grid.npz"
    with np.load(args.dataset_dir + fname) as f:
        df_points = f['sdf_points'] 
        filename = f['filename'] 
        vmax = f['vmax'] 
        vmin = f['vmin'] 
        importance_beta = f['beta']
    print(f"loaded {fname} shape: {df_points.shape}")
    '''
    
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