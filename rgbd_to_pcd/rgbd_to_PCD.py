# Libraries import
import open3d  as o3d
import numpy as np

spinholeCamera = o3d.io.read_pinhole_camera_intrinsic('camera_primesense.json')
pcds = []

# For each photo = each angle 
for i in range(506, 518):

    # Read the RGB and the depth images
    color_raw = o3d.io.read_image(fr'/home/user/scen_analyser/ImagesRGBCrop/Crop_rgb_42_{i}.jpg')
    depth_raw = o3d.io.read_image(fr'/home/user/scen_analyser/ImagesDepthCrop/Crop_depth_42_{i}.png')

    # Create the RGBD image from the 2 images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth( color_raw, depth_raw, convert_rgb_to_intensity=False )

    # Get the intrinsic settings of the camera
    camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("./d415.json")
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Create the point cloud from the RGBD image and the intrinsic settings
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    # Put the point cloud in the right way because it was backwards
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)
    o3d.io.write_point_cloud(fr"C:\Users\antoi\scen_analyser\3DPLY\out{i}.ply", pcds )

    # print(rgbd_image)
    # plt.subplot(1, 2, 1)
    # plt.title('Grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()


    #camera_extrinsec = np.array(0.999961, 0.00184501, -0.00861185, -0.00187475, 0.999992, -0.00344586, 0.00860543, 0.00346188, 0.999957)
    #camera_extrinsec.reshape(3, 3)

    



    # print(np.asarray(pcd.points))
    # print("\n")