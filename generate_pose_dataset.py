import os
import sys
import re
import numpy as np
import pandas as pd
import cv2
import glob
import json
import matplotlib.pyplot as plt

from predictors.posepredictor import PosePredictor

def main():
    pose_predictor = PosePredictor(num_threads=8)

    def check_or_create_dir(base_path, relative_path: str):
        dir_path = os.path.join(base_path, relative_path)
        if os.path.isdir(dir_path):
            print("The directory %s already exists" % dir_path)
        else:
            try:
                os.mkdir(dir_path)
            except  OSError:
                print("Creation of the directory %s failed" % dir_path)
            else:
                print("Successfully created the directory %s " % dir_path)
    
    with open("image_data_table.json") as image_talbe_file:
        image_data_table = json.load(image_talbe_file)

    X, y = [], []
    for dataset in image_data_table["datasets"]:
        check_or_create_dir(os.path.join(dataset["location"], dataset["version_num"]), "pose")
        for user in dataset["users"]:
            check_or_create_dir(os.path.join(dataset["location"], dataset["version_num"], "pose"), user)
            for date in dataset["dates"]:
                check_or_create_dir(os.path.join(dataset["location"], dataset["version_num"], "pose", user), date)
                for action in dataset["actions"]:
                    check_or_create_dir(os.path.join(dataset["location"], dataset["version_num"], "pose", user, date), action)
                    files_path = os.path.join(
                        dataset["location"],
                        dataset["version_num"],
                        dataset["type"],
                        user,
                        date,
                        action
                    )
                    save_path = os.path.join(
                        dataset["location"],
                        dataset["version_num"],
                        "pose",
                        user,
                        date,
                        action
                    )
                    # print(files_path)
                    times = glob.glob(os.path.join(files_path,"*"))
                    for time in times:
                        time_stamp = os.path.split(time)[-1]
                        check_or_create_dir(
                            os.path.join(
                                dataset["location"],
                                dataset["version_num"],
                                "pose", user, date, action
                                ), 
                            time_stamp
                            )
                        files = glob.glob(os.path.join(time,'*.png'))
                        # print(files)
                        for file in files:
                            file_name = os.path.basename(file).split('.')[0]
                            save_file = os.path.join(
                                dataset["location"],
                                dataset["version_num"],
                                "pose", user, date, action,
                                time_stamp, file_name + '.csv')
                            # if file already exist dont run tflite and just load it.
                            if os.path.isfile(save_file):
                                # load previously predicted pose
                                prediction = np.genfromtxt(save_file, delimiter=",")
                            else:
                                # load image and predict pose
                                image = cv2.imread(file, cv2.IMREAD_COLOR)
                                prediction = pose_predictor.predict([image])
                                # save pose data
                                np.savetxt(save_file, prediction, delimiter=",")
                                #print("image: ", file_name)
                                #print("save: ", save_file)
                            # add file to the data lists for exploration
                            X.append(prediction)
                            y.append(action)
    
    X_ndarray = np.array(X)
    print("X shape", X_ndarray.shape)
    y_ndarray = np.array(y)
    print("y shape", y_ndarray.shape)

if __name__ == "__main__":
    main()