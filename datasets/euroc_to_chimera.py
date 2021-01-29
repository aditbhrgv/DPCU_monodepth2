import os
from os.path import join
import cv2
import yaml
import numpy as np
import csv

eurc_sequece_dir = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/"
out_seq_path = "/home/adit/projects/data_recording_airsim/new"

if not os.path.isdir(out_seq_path):
    os.makedirs(out_seq_path)

eurc_image_folder = join(eurc_sequece_dir , "mav0/cam0/data")
eurc_imu_file = join(eurc_sequece_dir , "mav0/imu0/data.csv")
eurc_global_state_file = join(eurc_sequece_dir , "mav0/state_groundtruth_estimate0/data.csv")
camera_yaml_file = join(eurc_sequece_dir , "mav0/cam0/sensor.yaml")
output_yaml_file = join(out_seq_path , "sensor_calib.yaml")
output_intrinsics_file =  join(out_seq_path , "camera_intrinsics.csv")
images = os.listdir(eurc_image_folder)
images.sort()
img_tsts = [int(file.replace(".png","")) for file in images]

meta_fp = open(join(out_seq_path,"meta.csv"),"w")
imu_fp =  open(join(out_seq_path, "imu.csv"),"w")
gps_fp = open(join(out_seq_path, "gps.csv"),"w")
gt_fp = open(join(out_seq_path, "globalState.csv"),"w")

# Extract glob state data from csv file
globframes = []
globvalues = []
with open(os.path.join(eurc_global_state_file)) as csvfile:
    readglobstate = csv.reader(csvfile, delimiter=',')
    header = next(readglobstate)
    gval = list(readglobstate)
    for i in range(len(gval)):
        globframes.append(gval[i][0])
        globvalues.append(gval[i][1:])

    globvalues = np.array(globvalues)

glob_id_array = np.array(globframes)  # frameID of glob state
cam_id_array = np.array(img_tsts)

valid_glob_indices = np.nonzero(np.in1d(glob_id_array, cam_id_array))[0]
valid_cam_indices = np.nonzero(np.in1d(cam_id_array, glob_id_array))[0]

globvalues = globvalues[valid_glob_indices]

img_tsts = cam_id_array[valid_cam_indices]

time_scale = 1.0
img_index = 0

with open(eurc_imu_file) as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    header = next(readcsv)
    lines = list(readcsv)

    img_name = str(img_index).zfill(6)
    meta_fp.write(str(int(img_tsts[img_index] / time_scale)) + " " + img_name + "\n")

    glob_state_vals = globvalues[img_index]
    gt_fp.write(str(int(img_tsts[img_index] / time_scale)) + " " + img_name + " " + glob_state_vals[0] + " " +
                glob_state_vals[1] + " " + glob_state_vals[2] + " " + glob_state_vals[3] + " " +
                glob_state_vals[4] + " " + glob_state_vals[5] + " " + glob_state_vals[6]+ "\n")

    meta_fp.flush()
    ts_diff_camera = img_tsts[1] - img_tsts[0]
    for line in lines:
        curr_tst = int(line[0])
        ts_diff_imu_cam = (img_tsts[img_index] - curr_tst)
        # If there are more IMU frames captured before the camera recording was started save only IM
        # frames which belong to the first camera frame
        if img_index == 0 and ts_diff_imu_cam > ts_diff_camera:
            continue
        if curr_tst>img_tsts[img_index]:
            img_index += 1
            if len(img_tsts)==img_index:
                break
            img_name = str(img_index).zfill(6)
            meta_fp.write(str(int(img_tsts[img_index]/time_scale))+" "+img_name+"\n")
            meta_fp.flush()
            glob_state_vals = globvalues[img_index]

            gt_fp.write(str(int(img_tsts[img_index]/time_scale)) + " " + img_name + " " + glob_state_vals[0] + " " +
                        glob_state_vals[1] + " " + glob_state_vals[2] + " " + glob_state_vals[3] + " " +
                        glob_state_vals[4] + " " + glob_state_vals[5] + " " + glob_state_vals[6] + "\n")

            gt_fp.flush()

        img_name = str(img_index).zfill(6)
        imu_fp.write(str(int(curr_tst/time_scale))+" "+img_name+" "+line[4]+" "+line[5]+" "+line[6]+" "+line[1]+" "+line[2]+" "+line[3]+"\n")
        imu_fp.flush()
        #Dummy GPS file for our dataset format as EuroC doesn't provide it
        gps_fp.write(str(int(curr_tst/time_scale))+" "+img_name+" "+"48.0 -100.0 100.0\n")
        gps_fp.flush()

meta_fp.close()
imu_fp.close()
gps_fp.close()
gt_fp.close()


img = cv2.imread(join(eurc_image_folder, str(img_tsts[0])+".png" ))
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
roi_list[2] = 704
roi_list[3] = 384
roi = tuple(roi_list)

with open(output_intrinsics_file, "w") as csvfile:
    writecsv = csv.writer(csvfile, delimiter=' ')
    writecsv.writerow(newcameramtx.flatten())

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist_coeff,None,newcameramtx,(w,h),5)
index = 0

if not os.path.isdir(join(out_seq_path, "camera")):
    os.makedirs(join(out_seq_path, "camera"))

with open(output_yaml_file, 'w') as out_file:
    with open(camera_yaml_file, 'r') as fp:
        yaml_data_loaded = yaml.safe_load(fp)
        T_body2cam = yaml_data_loaded['T_BS']['data']
        data = {'T_BODY_CAMERA': T_body2cam, 'T_BODY_IMU': np.eye(4).tolist()}
        yaml.dump(data, out_file)


for image in img_tsts:
    img = cv2.imread(join(eurc_image_folder, str(image)+ ".png"))
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(join(join(out_seq_path, "camera"), str(index).zfill(6)+".png"), dst)
    index += 1



