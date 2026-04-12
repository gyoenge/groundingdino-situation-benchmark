"""
    origin label, image 폴더 내 파일들에 대해, 파일명에 key가 포함된 파일들을 copy_path로 이동
    - 폴더 경로 정보가 origin_pathes = [path_a, path_b, ...] 일 때 
    - 각각의 경로에 해당하는 폴더 안에서, 파일명에 "_c_"가 포함되는 파일들을 모두 copy_path 로 복사
"""

import os
import shutil

# path settings 
label_origin_rootpath = "D:/net_dataset/173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터/01.데이터/1.Training/라벨링데이터/TL_행위(불법행위)데이터1/1.불법행위/1.흡연행위"
video_origin_rootpath = "D:/net_dataset/173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터/01.데이터/1.Training/원천데이터/TS_행위(불법행위)데이터1/1.불법행위/1.흡연행위"
def get_subdirectories(main_path):
    subdirs = []
    for dirpath, dirnames, filenames in os.walk(main_path):
        for dirname in dirnames:
            subdirs.append(os.path.join(dirpath, dirname))
    return subdirs
label_origin_pathes = get_subdirectories(label_origin_rootpath)
label_copy_path = "raw_data/label"
video_origin_pathes = get_subdirectories(video_origin_rootpath)
video_copy_path = "raw_data/video"
detect_key = "_c_"

print("=============================================================================")
def delete_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
print(f"** '{label_copy_path}', '{video_copy_path}' 폴더 초기화")
delete_files_in_directory(label_copy_path)
delete_files_in_directory(video_copy_path)
print("=============================================================================")
for label_origin_path in label_origin_pathes:
    print(f"============={os.path.basename(label_origin_path)}=============")
    for file in os.listdir(label_origin_path):
        if "_c_" in file:
            print(f"Copy label file '{os.path.basename(file)}'")
            src = os.path.join(label_origin_path, file)
            dst = os.path.join(label_copy_path, file)

            # 파일 복사. 이미 존재하는 경우 덮어씀.
            shutil.copy2(src, dst)
print("Labels all copied")
print("=============================================================================")
for video_origin_path in video_origin_pathes:
    print(f"============={os.path.basename(video_origin_path)}=============")
    for file in os.listdir(video_origin_path): 
        if "_c_" in file:
            print(f"Copy video file '{os.path.basename(file)}'")
            src = os.path.join(video_origin_path, file)
            dst = os.path.join(video_copy_path, file)

            # 파일 복사. 이미 존재하는 경우 덮어씀.
            shutil.copy2(src, dst)
print("Videos all copied")   
print("=============================================================================")

