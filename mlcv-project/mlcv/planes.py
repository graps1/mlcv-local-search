import numpy as np
import pandas as pd

def cf(data, u,v,w):
    return 0

def cf_prime(data,u,v,w,dthr=0.1):
    # this could be much faster if we didn't use numpy
    # compute distance between plane created by points u,v,w and origin (0,0,0)
    points = data.loc[[u,v,w],["x","y","z"]].to_numpy()
    n = np.cross(points[1,:] - points[0,:], points[2,:] - points[0,:])
    d = abs(n.dot(points[0,:])/np.linalg.norm(n))
    # returns d-dthr if d>dthr and 0 otherwise
    return max(d-dthr, 0)

def draw_points_from_plane(nr_points, normal_vector, orientation, noise):
    points2d = np.random.uniform(-1,1, size=(nr_points,2,1))
    noise = np.random.uniform(-noise,noise,size=(nr_points,1))
    third_axis = np.cross(normal_vector, orientation)
    third_axis = third_axis/np.linalg.norm(third_axis)
    broadcasted_x = np.repeat(orientation, nr_points).reshape(3,nr_points).T
    broadcasted_y = np.repeat(third_axis, nr_points).reshape(3,nr_points).T
    broadcasted_z = np.repeat(normal_vector, nr_points).reshape(3,nr_points).T
    points3d = points2d[:,0]*broadcasted_x + \
               points2d[:,1]*broadcasted_y + \
               noise*broadcasted_z
    return points3d

def generate_points(
        noise = [0.2, 0.2, 0.2, 0.2],
        distribution=[400, 400, 400, 400]):
    """Generates a set of points drawn from random planes.

    :param noise: The added noise/plane, defaults to [0.2, 0.2, 0.2, 0.2]
    :type noise: list, optional
    :param distribution: The points/plane distribution, defaults to [400, 400, 400, 400]
    :type distribution: list, optional
    :return: A DataFrame with columns x,y,z,label
    :rtype: pandas.DataFrame
    """    

    nr_planes = len(distribution)

    # for each plane, randomly sample a normal vector
    # by drawing uniformly random points from a sphere
    # source https://www.jasondavies.com/maps/random-points/
    lambdas = np.random.uniform(-np.pi,np.pi,size=2*nr_planes)
    xs = np.random.uniform(0,1,size=2*nr_planes)
    phis = np.arccos(2*xs-1)
    coordinates = np.vstack((
        np.sin(phis)*np.cos(lambdas),
        np.sin(phis)*np.sin(lambdas),
        np.cos(phis)))
    normal_vectors = coordinates[:,:nr_planes]
    orientations = np.cross(normal_vectors, coordinates[:,nr_planes:], axisa=0, axisb=0, axisc=0)
    orientations = orientations/np.linalg.norm(orientations, axis=0)[None,:]

    # draw random points for each plane
    df = pd.DataFrame()
    for idx, pointcount in enumerate(distribution):
        # draw points from R^2 with noise and compute their
        # position on the plane
        samples = draw_points_from_plane(
            pointcount,
            normal_vectors[:,idx],
            orientations[:,idx],
            noise[idx])
        df = pd.concat([df, pd.DataFrame(
            { "x": samples[:,0],
              "y": samples[:,1],
              "z": samples[:,2],
              "partition": np.ones(pointcount, dtype=np.int)*idx }) ],
              ignore_index=True)

    return df