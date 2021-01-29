#Imports for conversion file :
# File: euroC to Chimera data format conversion
import os
from os.path import join
import cv2
import yaml
import numpy as np
import csv

#Paths to the folders to read and save data to !
#Please change the paths according to your own local paths

# Root path of the EuroC sequence to be converted
euroc_sequence_root_path = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/"
# Root path of the converted Chimera data format sequence which needs to be generated 
chimera_sequence_root_path = "/home/adit/projects/data_recording_airsim/new_sequence"
# Create a new root folder for Chimera sequence to be generated if not exists already
if not os.path.isdir(chimera_sequence_root_path):
    os.makedirs(chimera_sequence_root_path)
if not os.path.isdir(join(chimera_sequence_root_path, "camera")):
    os.makedirs(join(chimera_sequence_root_path, "camera"))

#EuroC sensor paths which is relative to euroc_sequence_root_path
euroc_image_folder = join(euroc_sequence_root_path , "mav0/cam0/data")
euroc_imu_file = join(euroc_sequence_root_path , "mav0/imu0/data.csv")
euroc_global_state_file = join(euroc_sequence_root_path , "mav0/state_groundtruth_estimate0/data.csv")
camera_yaml_file = join(euroc_sequence_root_path , "mav0/cam0/sensor.yaml")

#Chimera storage paths relative to chimera_sequence_root_path

#Sensor calibration file to store EuroC's Body2Camera and Body2Imu transformation at generated Chimera sequence root folder 
chimera_sensor_calib_file = join(chimera_sequence_root_path , "sensor_calib.yaml")
#Camera intrinsics file to store the EuroC camera intrincs at generated Chimera sequence root folder 
chimera_camera_intrinsics_file = join(chimera_sequence_root_path , "camera_intrinsics.csv")
#Meta file,imu,gps and globalstate files to be stored at generated Chimera sequence root folder 
chimera_meta_file = open(join(chimera_sequence_root_path,"meta.csv"),"w")
chimera_imu_file =  open(join(chimera_sequence_root_path, "imu.csv"),"w")
chimera_gps_file = open(join(chimera_sequence_root_path, "gps.csv"),"w")
chimera_glob_state_file = open(join(chimera_sequence_root_path, "globalState.csv"),"w")

#Load images from EuroC images folder
images = os.listdir(euroc_image_folder)
images.sort()
#Extract the image timestamps
img_timestamps = [int(file.replace(".png","")) for file in images]

# Extract global state data from Euroc csv file
globalstate_timestamps = []
globvalues = []
with open(os.path.join(euroc_global_state_file)) as csvfile:
    readglobstate = csv.reader(csvfile, delimiter=',')
    header = next(readglobstate)
    gval = list(readglobstate)
    for i in range(len(gval)):
        globalstate_timestamps.append(gval[i][0])
        globvalues.append(gval[i][1:])
    globvalues = np.array(globvalues)

glob_id_array = np.array(globalstate_timestamps)  #frameID present in glob state
cam_id_array = np.array(img_timestamps) #frameID of the camera

#Find valid indices for which globalstate and camera values both exist
valid_glob_indices = np.nonzero(np.in1d(glob_id_array, cam_id_array))[0]
valid_cam_indices = np.nonzero(np.in1d(cam_id_array, glob_id_array))[0]

#Extract valid globalstate and camera frames/timestamp values
globvalues = globvalues[valid_glob_indices]
img_timestamps = cam_id_array[valid_cam_indices]

time_scale = 1.0
img_index = 0
with open(euroc_imu_file) as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    header = next(readcsv)
    lines = list(readcsv)

    #Special handling for the 0th frame for meta and globalstate output files
    img_name = str(img_index).zfill(6)
    chimera_meta_file.write(str(int(img_timestamps[img_index] / time_scale)) + " " + img_name + "\n")
    glob_state_vals = globvalues[img_index]
    chimera_glob_state_file.write(str(int(img_timestamps[img_index] / time_scale)) + " " + img_name + " " + glob_state_vals[0] + " " +
                glob_state_vals[1] + " " + glob_state_vals[2] + " " + glob_state_vals[3] + " " +
                glob_state_vals[4] + " " + glob_state_vals[5] + " " + glob_state_vals[6]+ "\n")
    chimera_meta_file.flush() #Flush the file pointer currently as sometimes all the information is not written !
                              #TODO: Find better way!
    ts_diff_camera = img_timestamps[1] - img_timestamps[0]
    for line in lines:
        curr_tst = int(line[0])
        ts_diff_imu_cam = (img_timestamps[img_index] - curr_tst)
        # If there are more IMU frames captured before the camera recording was started save only IMU
        # frames which belong to the first camera frame
        if img_index == 0 and ts_diff_imu_cam > ts_diff_camera:
            continue
        if curr_tst>img_timestamps[img_index]:
            img_index += 1
            if len(img_timestamps)==img_index:
                break
            img_name = str(img_index).zfill(6)
            chimera_meta_file.write(str(int(img_timestamps[img_index]/time_scale))+" "+img_name+"\n")
            chimera_meta_file.flush()
            glob_state_vals = globvalues[img_index]

            chimera_glob_state_file.write(str(int(img_timestamps[img_index]/time_scale)) + " " + img_name + " " + glob_state_vals[0] + " " +
                        glob_state_vals[1] + " " + glob_state_vals[2] + " " + glob_state_vals[3] + " " +
                        glob_state_vals[4] + " " + glob_state_vals[5] + " " + glob_state_vals[6] + "\n")
            chimera_glob_state_file.flush()

        img_name = str(img_index).zfill(6)
        chimera_imu_file.write(str(int(curr_tst/time_scale))+" "+img_name+" "+line[4]+" "+line[5]+" "+line[6]+" "+line[1]+" "+line[2]+" "+line[3]+"\n")
        chimera_imu_file.flush()
        #Dummy GPS file for our dataset format as EuroC doesn't provide it
        chimera_gps_file.write(str(int(curr_tst/time_scale))+" "+img_name+" "+"48.0 -100.0 100.0\n")
        chimera_gps_file.flush()

#Close all the files once writing is finished
chimera_meta_file.close()
chimera_imu_file.close()
chimera_gps_file.close()
chimera_glob_state_file.close()

# Read and Prepare to undistort the EuroC images
img = cv2.imread(join(euroc_image_folder, str(img_timestamps[0])+".png" ))
h,  w = img.shape[:2]
mtx = np.eye(3)
dist_coeff = []
with open(camera_yaml_file, 'r') as fp:
    yaml_data_loaded = yaml.safe_load(fp)
    intrinsics_list = yaml_data_loaded['intrinsics']
    dist_coeff = np.array(yaml_data_loaded['distortion_coefficients'])
    mtx[0, 0] =  intrinsics_list[0]
    mtx[1, 1] = intrinsics_list[1]
    mtx[0, 2] = intrinsics_list[2]
    mtx[1, 2] = intrinsics_list[3]

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist_coeff,(w,h),1,(w,h))
roi_list =  list(roi)
roi_list[2] = 704 #Values set as multiple of 32 was required by Monodepth2 training
roi_list[3] = 384 #Values set as multiple of 32 was required by Monodepth2 training
roi = tuple(roi_list)

#Write new camera matrix intrincics after undistortion at Chimera root path
with open(chimera_camera_intrinsics_file, "w") as csvfile:
    writecsv = csv.writer(csvfile, delimiter=' ')
    writecsv.writerow(newcameramtx.flatten())

# undistort the images
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist_coeff,None,newcameramtx,(w,h),5)

#Write sensor calib file at Chimera root path
with open(chimera_sensor_calib_file, 'w') as out_file:
    with open(camera_yaml_file, 'r') as fp:
        yaml_data_loaded = yaml.safe_load(fp)
        T_body2cam = yaml_data_loaded['T_BS']['data']
        data = {'T_BODY_CAMERA': T_body2cam, 'T_BODY_IMU': np.eye(4).tolist()}
        yaml.dump(data, out_file)

#Copy the undistorted EuroC images at the Chimera_root_dir/camera folder
index = 0
for image in img_timestamps:
    img = cv2.imread(join(euroc_image_folder, str(image)+ ".png"))
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(join(join(chimera_sequence_root_path, "camera"), str(index).zfill(6)+".png"), dst)
    index += 1



