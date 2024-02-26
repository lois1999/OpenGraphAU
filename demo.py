import os
import numpy as np
import torch
import torch.nn as nn
import logging
from dataset import pil_loader
from model.ANFL import MEFARG
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
import csv

def main(conf):
    dataset_info = hybrid_prediction_infolist

    # data
    folder_path = conf.input

    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    csv_file_path = "predictions_AU.csv"

    i = 0
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".jpg"):
                    print("Working on {}...".format(file))
                    file_path = os.path.join(root, file)
                    new_root, pair_part_turn = os.path.split(root)
                    participant_id = (pair_part_turn).split("_")[1][-1]
                    turn = (pair_part_turn).split("_")[2][-1]
                    new_root, pair = os.path.split(new_root)
                    pair_id = pair[2][-1]
                    new_root, condition_id = os.path.split(new_root)
                    condition = condition_id[1][-1]
                 
                    net.eval()
                    img_transform = image_eval()
                    img = pil_loader(file_path)
                    img_ = img_transform(img).unsqueeze(0)

                    if torch.cuda.is_available():
                        net = net.cuda()
                        img_ = img_.cuda()

                    with torch.no_grad():
                        pred = net(img_)
                        pred = pred.squeeze().cpu().numpy()

                    infostr_probs, infostr_aus = dataset_info(pred, 0.5)
                        
                    au_elements = str(infostr_probs).strip("{}'").split()
                    
                    au_str = ""
                    for j in range(0, len(au_elements), 2):
                        au_str += f"{au_elements[j]} {au_elements[j+1]},"

                    pairs = au_str.split(",")

                    au_dict = {item[0]: float(item[-1]) for item in (pair.split(': ') for pair in pairs[:-1])}
                    
                    # header
                    if i == 0:
                        header = ["condition", "pair_id", "participant_id", "turn"] + list(au_dict.keys())
                        csvwriter.writerow(header)

                    i += 1
                    data_row = [condition, pair_id, participant_id, turn] + list(au_dict.values())
                    csvwriter.writerow(data_row)

                    # log
                    infostr = {'AU prediction:'}
                    logging.info(infostr)
                    infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
                    logging.info(infostr_aus)
                    logging.info(infostr_probs)

                    if conf.draw_text:
                        img = draw_text(file_path, list(infostr_aus), pred)
                        import cv2
                        path = conf.input.split('.')[0]+'_pred.jpg'
                        cv2.imwrite(path, img)


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

