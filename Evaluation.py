# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from config.Config import Config
# ============= Dataset =====================
from lib.dataset.collate import collate_single_cpu
from lib.dataset.dataset_for_argoverse import STFDataset as ArgoverseDataset
# ============= Models ======================
from lib.models.mmTransformer import mmTrans
from lib.utils.evaluation_utils import compute_forecasting_metrics, FormatData

# from lib.utils.traj_nms import traj_nms
from lib.utils.utilities import load_checkpoint, load_model_class

import csv

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model-name', type=str, default='demo')
    parser.add_argument('--model-save-path', type=str, default='./models/')
    parser.add_argument('--vehicle', type=str, default="AGENT")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    print("gpu number:{}".format(gpu_num))

    args = parse_args()
    cfg = Config.fromfile(args.config)

    # ================================== INIT DATASET ==========================================================
    validation_cfg = cfg.get('val_dataset')
    validation_cfg['processed_data_path'] = os.path.join(validation_cfg['processed_data_path'], args.vehicle, "argoverse_info_val.pkl")
    validation_cfg['processed_maps_path'] = os.path.join(validation_cfg['processed_maps_path'], args.vehicle, "map.pkl")
    val_dataset = ArgoverseDataset(validation_cfg)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=validation_cfg["shuffle"],
                                batch_size=validation_cfg["batch_size"],
                                num_workers=validation_cfg["workers_per_gpu"],
                                collate_fn=collate_single_cpu)
    # =================================== Metric Initial =======================================================
    format_results = FormatData()
    evaluate = partial(compute_forecasting_metrics,
                       max_n_guesses=6,
                       horizon=30,
                       miss_threshold=2.0)
    # =================================== INIT MODEL ===========================================================
    model_cfg = cfg.get('model')
    stacked_transfomre = load_model_class(model_cfg['type'])
    model = mmTrans(stacked_transfomre, model_cfg).cuda() if gpu_num > 0 else  mmTrans(stacked_transfomre, model_cfg)
    model_name = os.path.join(args.model_save_path,
                              '{}.pt'.format(args.model_name))
    model = load_checkpoint(model_name, model)
    print('Successfully Loaded model: {}'.format(model_name))
    print('Finished Initialization in {:.3f}s!!!'.format(
        time.time()-start_time))
    # ==================================== EVALUATION LOOP =====================================================
    model.eval()
    progress_bar = tqdm(val_dataloader)
    i = 0
    with torch.no_grad():
        with open('latency.csv', 'w', newline='') as csvfile:
            latencywriter = csv.writer(csvfile, delimiter=' ')
            for j, data in enumerate(progress_bar):
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda() if gpu_num > 0 else data[key]
                t = time.time()
                out = model(data)
                latencywriter.writerow({time.time() - t})
                format_results(data, out) # incrementally populated

    # Output results to csv files
    dir = os.path.join('./data/results/', args.vehicle)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for key, value in format_results.results['forecasted_trajectories'].items():
        with open(os.path.join(dir, key+'.csv'), 'w', newline='') as csvfile:
            temp = str(float(val_dataset.get_data(key)["NORM_CENTER"][0])) + "," + str(float(val_dataset.get_data(key)["NORM_CENTER"][1]))+",|"
            for i in range(len(value)): #6
                for j in range(0,len(value[0])): #30
                    temp = temp + "," + str(value[i][j][0]) + "," + str(value[i][j][1])
                temp = temp + ",|"
            temp = temp + ","
            csvfile.write(temp)
        with open(os.path.join(dir, key+'_prob.csv'), 'w', newline='') as csvfile:
            prob = format_results.results['forecasted_probabilities'][key]
            temp = "1.0,|,"
            for i in range(len(value)): #6
                temp = temp + str(prob[i]) + ",|,"
            temp = temp + "\n"
            csvfile.write(temp)


    print(evaluate(**format_results.results))
    print('Validation Process Finished!!')
