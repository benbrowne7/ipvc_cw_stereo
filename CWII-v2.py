'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from numpy import array_equal
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')
    parser.add_argument('--visual', dest='bVis', action='store_true')
    parser.add_argument('--corres', dest='corres_search', type=str, default='epipolar',
                        help='method for correspondence searching, choose from [epipolar, colour]')
    parser.add_argument('--find_sphere', dest='bDepth', action='store_true')
    args = parser.parse_args()

    img_width = 640
    img_height = 480

    num = args.num


    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sephere with random size
        size = random.randrange(10, 14, 2) / 10.
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # random set sphere location
        step = 6
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))
        # print(f'sphere_{i}: [{size}, {x}, {z}] {size}')
        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    print("theta:", theta)
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    # set camera intrinsics
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 415.69219381653056, 415.69219381653056, 319.5, 239.5)
    print('Camera parameters:')
    print('Pose_0\n', H0_wc)
    print('Pose_1\n', H1_wc)
    print('Intrinsics\n', K.intrinsic_matrix)
    # o3d.io.write_pinhole_camera_intrinsic("test.json", K)


    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    
    ###################################
    '''
    Question 3: Circle detection
    Hint: check cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    
    #for both images, run hough circle detection, draw circles onto image and output
    for x in range(1,-1, -1):
        circles = []
        frame = cv2.imread("view" + str(x) + ".png", )
        output = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5,5), 0)
        circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=30, param2=32, minRadius=10, maxRadius=50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        j = 0
        for i in circles[0,:]:
            xc = int(i[0])
            yc = int(i[1])
            r = int(i[2])
            cv2.circle(output, (xc,yc), r, (0,255,0), 2)
            if x==1:
                output[yc,xc] = (0,0,255)
                cv2.putText(output, str(j), (xc+4,yc), font, 0.4, color=(255,0,0), thickness=2)
                j += 1
                #cv2.circle(output, (xc,yc), 1, (0,0,254), 1)
            if x==0:
                output[yc,xc] = (255,0,254)
                output[yc-1,xc] = (255,0,254)
                output[yc-1,xc-1] = (255,0,254)
                output[yc-1,xc+1] = (255,0,254)
                output[yc+1,xc] = (255,0,254)
                output[yc+1,xc-1] = (255,0,254)
                output[yc,xc-1] = (255,0,254)
                output[yc+1,xc+1] = (255,0,254)
                output[yc,xc+1] = (255,0,254)
                #cv2.circle(output, (xc,yc), 1, (0,0,254), 1)
        cv2.imwrite("circles" + str(x) + ".png", output)

    ###################################
    '''
    Question 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################
    def camera_part(x):
        pos = x[:, 3]
        pos = np.delete(pos, 3, axis=0)
        rmat = np.delete(x, 3, axis=0)
        rmat = np.delete(rmat, 3, axis=1)
        return rmat, pos
   
    H0_cw = np.linalg.inv(H0_wc)
    H1_cw = np.linalg.inv(H1_wc)

    camleft_Rmat, camleft_vec = camera_part(H1_wc)
    camright_Rmat, camright_vec = camera_part(H0_wc)
 
    #PL = camleft_Rmat.dot(camleft_vec.T)
    #PR = camright_Rmat.dot(camright_vec.T)

    HRL = np.matmul(H1_wc, np.linalg.inv(H0_wc))
    R_trans, T = camera_part(HRL)
    R = R_trans.T

    S = [[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]]
    E = np.matmul(R, S)
    
    M = np.linalg.inv(K.intrinsic_matrix)
    F = np.matmul(M.T, np.matmul(E, M))

    #take left camera/H1 as our reference view
    ref_image = cv2.imread("circles1.png", )
    other_image = cv2.imread("circles0.png", )
    other_image_orig = other_image.copy()
    img_height, img_width = ref_image.shape[:2]

    ref_centers = []

    #searching for circle centers in reference image and filter duplicates
    for x in range(0, img_width):
        for y in range(0, img_height):
            (b, g, r) = ref_image[y,x]
            if (r == 255) and (g == 0) and (b == 0):
                ref_centers.append(np.array([x,y,1]))
    #print("Ref Centers:", ref_centers)
    ref_centers = np.array(ref_centers)
    

    #calculate corresponding epipolar lines in other image (same order as circles detected)
    corres_epipolars = []
    for cent in ref_centers:
        cent = np.array(cent)
        line = np.dot(F, cent.T)
        corres_epipolars.append(line)
    corres_epipolars = np.array(corres_epipolars)
    #print(corres_epipolars)

    #draw epipolar lines in other image
    for i in range(0, len(ref_centers)):
        (a,b,c) = corres_epipolars[i]
        x = int(round(-c/a))
        start = (x,0)
        y = img_height
        x1 = int(round((-c - b*y) / a))
        end = (x1, img_height)
        cv2.line(other_image, start, end, (255,0,0), 1)
    cv2.imwrite("corres_epilines.png", other_image)
    

    ###################################
    '''
    Question 5: Find correspondences

    Write your code here
    '''
    ###################################

    #search along each epipolar line to find corresponding circle center, tolerance of +/-2 y coord
    corres_centers = []
    fresh = cv2.imread("circles0.png", )

    def duplicate_check(candidate, corres_centers):
      return next((True for elem in corres_centers if array_equal(elem, corres_centers)), False)

    for line in corres_epipolars:
        (a1,b1,c1) = line
        #print(line)
        for x in range(0, img_width):
            y = round((-c1 - a1*x) / b1)
            candidate = [x,y,1]
            if y < 0:
                continue
            if y > img_height-10:
                continue
            (b,g,r) = fresh[y,x]
            if (r == 254):
                if candidate in corres_centers:
                    continue
                corres_centers.append(candidate)
                break
            (b,g,r) = fresh[y-1,x]
            if (r == 254):
                if candidate in corres_centers:
                    continue
                corres_centers.append(candidate)
                break
            (b,g,r) = fresh[y+1,x]
            if (r == 254):
                if candidate in corres_centers:
                    continue
                corres_centers.append(candidate)
                break
            (b,g,r) = fresh[y-2,x]
            if (r == 254):
                if candidate in corres_centers:
                    continue
                corres_centers.append(candidate)
                break
            (b,g,r) = fresh[y+2,x]
            if (r == 254):
                if candidate in corres_centers:
                    continue
                corres_centers.append(candidate)
                break
    #print(np.array(corres_centers))

    for cent in corres_centers:
        xc, yc, scrap = cent
        cv2.putText(other_image, "x", (xc+4,yc), font, 0.4, color=(254,0,0), thickness=2)
        j +=1
    cv2.imwrite("corres_epilines.png", other_image)


    ###################################
    '''
    Question 6: 3-D locations of spheres

    Write your code here
    '''
    ###################################

    #check if all centers were found
    if (len(ref_centers) != len(corres_centers)):
        print("not all corresponding points were found: try again")
        raise SystemExit(0)
    
    #get matching point pairs 
    point_pairs = []
    for pl in ref_centers:
        for pr in corres_centers:
            pr = np.array(pr)
            g = np.dot(pr.T, np.matmul(F, pl))
            if (-0.1 < g < 0.1):
                point_pairs.append((pl, pr))
    if (len(point_pairs) != num):
        print("could not match all point pairs: try again")
        raise SystemExit(0)


    #calculating H matrix and corresponding point in 3D space for each pair
    for pair in point_pairs:
        pl, pr = pair
        col1 = pl.T
        col2 = np.matmul(R.T, pr)
        col3 = np.cross(pl, np.matmul(R.T, pr))    
        h = np.append([col1, col2], [col3], axis=0)
        H = h.T
        abc = np.matmul(np.linalg.inv(H), T)
        a, b, c = abc
        phat = (a*pl + b*(np.matmul(R.T, pr)) + T) / 2
        print(phat)
        #small c means a good correspondance 

    vis.run()
    vis.destroy_window()


    ###################################
    '''
    Question 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Question 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Question 9: Display the spheres

    Write your code here:
    '''
    ###################################
