# import csv, codecs
#
# annotation_path = "/data/dataset/zhb/csv/gpt_all_new.csv"
# train_path = "/data/dataset/zhb/csv/train.csv"
# val_path = "/data/dataset/zhb/csv/test.csv"
#
#
# train_file = open(train_path, 'w', encoding='utf-8', newline='')
# val_file = open(val_path, 'w', encoding='utf-8', newline='')
# train_writer = csv.writer(train_file)
# train_writer.writerow(
#     [f'pic_name', f'score', f'score_shape', f'score_pose', f'preference_score_list', f'bmi'])
#
# val_writer = csv.writer(val_file)
# val_writer.writerow(
#     [f'pic_name', f'score', f'score_shape', f'score_pose', f'preference_score_list', f'bmi'])
#
# max_score = 10
# val_max = 7042
# val_num_list = []
# all_max = 37042
# split = float(all_max / val_max)
# l = 0.0
# r = 37042.0
# for i in range(0, all_max):
#     l += split
#     r -= split
#     if r <= l:
#         break
#     if len(val_num_list) + 2 <= val_max:
#         if l == r:
#             val_num_list.append(round(l))
#         else:
#             val_num_list.append(round(l))
#             val_num_list.append(round(r))
# index = -1
# with codecs.open(annotation_path, encoding='utf-8-sig', errors='ignore') as f:
#     for row in csv.reader(f, skipinitialspace=True):
#         if index < 0:
#             index = 0
#             continue
#         if len(row) == 0:
#             continue
#         if index in val_num_list:
#             val_writer.writerow(row)
#         else:
#             train_writer.writerow(row)
#         index += 1


"""修改权重"""

# import os
# import torch

# from models_.physique_frame import PhysiqueFrame


# model = PhysiqueFrame()


# load_pth_path = (
#     "/data/dataset/zhb/PhysiqueFrame_pth/User-ISTJ/"
#     + "PAA-224px_24_7_8_black224_vacc0.7163509471585244_srcc0.6165534524668317vlcc0.6499156231970553_vaccS0.726321036889332_srccS0.5458478916123467vlccS0.569646271311982_vaccP0.7008973080757727_srccP0.5972757110028716vlccP0.6408747693442073_1.pth"
# )
# state_dict = torch.load(load_pth_path, map_location="cuda:0")

# new_state_dict = {}
# model_dict = model.state_dict()


# for k, v in state_dict.items():
#     if k in model_dict.keys():
#         new_state_dict[k] = v
#     elif "pam_classifier" in k:
#         k = k.replace("pam_classifier", "appearance_classifier")
#         new_state_dict[k] = v
#     elif "pam_shape_classifier" in k:
#         k = k.replace("pam_shape_classifier", "health_classifier")
#         new_state_dict[k] = v
#     elif "pam_pose_classifier" in k:
#         k = k.replace("pam_pose_classifier", "posture_classifier")
#         new_state_dict[k] = v


# model.load_state_dict(new_state_dict)

# torch.save(
#     new_state_dict,
#     "/home/zhb/code/PAA/AVA_Swin/pth/train/final/"
#     + "PhysiqueFrame_"
#     + "ISTJ_"
#     + "vaccA0.7163509471585244_srccA0.6165534524668317_vlccA0.6499156231970553_vaccH0.726321036889332_srccH0.5458478916123467_vlccH0.569646271311982_vaccP0.7008973080757727_srccP0.5972757110028716_vlccP0.6408747693442073.pth",
# )

""" 修改数据集 """

# import os
# import pandas as pd
# import shutil

# # Define paths
# csv_folder_path = "/data/dataset/zhb/PhysiqueAA50K_csv/pretrain"
# output_folder_path = "/data/dataset/zhb/github/csv/pretrain"

# # Walk through the directory and process csv files
# for root, dirs, files in os.walk(csv_folder_path):
#     for file in files:
#         # Check if the file is a CSV and is not "test.csv"
#         if file.endswith(".csv") and file != "test.csv":
#             # Construct the full file path
#             file_path = os.path.join(root, file)

#             # Read the CSV file into a DataFrame
#             df = pd.read_csv(file_path)

#             # Rename the columns
#             df.columns = [
#                 "pic_name",
#                 "score_appearance",
#                 "score_health",
#                 "score_posture",
#                 "preference_score_list",
#                 "bmi",
#             ]
#             if "ISFJ" in file_path:
#                 df["people_answer"] = "[1, 1, 1]"
#             elif "ESFJ" in file_path:
#                 df["people_answer"] = "[1, 1, -1]"
#             elif "ISTJ" in file_path:
#                 df["people_answer"] = "[-1, -1, 1]"

#             # Define the output path
#             output_file_path = os.path.join(
#                 output_folder_path, os.path.relpath(root, csv_folder_path), file
#             )

#             # Ensure the output directory exists
#             os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

#             # Save the modified CSV to the new location
#             df.to_csv(output_file_path, index=False)
