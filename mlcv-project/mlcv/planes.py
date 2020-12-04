import numpy as np


def distance_to_points(data,points,combs):
    """for each 3-ary vertex-pairing, computes the matching plane and its distance
    to a given set of points.

    :param data: the dataset containing 3d-points for n vertices
    :type data: np.array[n,3,dtype=float]
    :param combs: selection of m 3-ary subsets of vertices
    :type combs: np.array[m,3,dtype=int]
    :param points: array containing m times 3d-points
    :type points: np.array[m,3,dtype=float]
    :return: an array containing the distances for each entry in combs
    :rtype: np.array[m,dtype=float]
    """    
    # compute distance between plane created by points u,v,w and origin (0,0,0)
    planes = np.stack(( data[combs[:,0]],
                        data[combs[:,1]],
                        data[combs[:,2]] ))
    # points.shape == (#comb-len=3, #combinations, #pointdimension=3)
    # compute normal vectors -> shape == (#combinations, #pointdimension=3)
    n = np.cross(planes[1,:,:] - planes[0,:,:], planes[2,:,:] - planes[0,:,:])
    # computes multiple dot products at once 
    # -> shape == (#combinations, 1)
    dots = np.einsum("ij,ij->i",n,planes[0,:,:]-points)
    # computes the distances by normalizing dot products
    # -> shape == (#combinations, 1)
    d = np.abs(dots/np.linalg.norm(n,axis=1))
    return d


def distance_to_origin(data,combs):
    """for each 3-ary vertex-pairing, computes the matching plane and its distance
    to the coordinate origin.

    :param data: the dataset containing 3d-points for n vertices
    :type data: np.array[n,3,dtype=float]
    :param combs: selection of m 3-ary subsets of vertices
    :type combs: np.array[m,3,dtype=int]
    :return: an array containing the distances for each entry in combs
    :rtype: np.array[m,dtype=float]
    """
    return distance_to_points(data, np.zeros(combs.shape), combs)


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
    data, partition = None, None
    for idx, pointcount in enumerate(distribution):
        # draw points from R^2 with noise and compute their
        # position on the plane
        samples = draw_points_from_plane(
            pointcount,
            normal_vectors[:,idx],
            orientations[:,idx],
            noise[idx])
        if data is None:
            data = samples
            partition = np.ones(pointcount, dtype=np.int)*idx
        else:
            data = np.vstack((data, samples))
            partition = np.hstack((partition, 
                                   np.ones(pointcount, dtype=np.int)*idx))

    return data, partition