import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [14, 8]
import motmetrics as mm
from motmetrics.distances import iou_matrix
import pandas as pd

sys.path.append('../..')
from pytracking.utils.load_text import load_text
from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pprint import pprint

tkl = []
pt = '../tracking_results'
for trk in os.listdir(pt):
    for param in os.listdir(os.path.join(pt, trk)):
        tkl.append([trk, param, trackerlist(trk, param, None)[0]])
dataset = get_dataset('yt_pb_valid')

def motMetricsEnhancedCalculator(gt, t):
  # import required packages
  
  acc = mm.MOTAccumulator(auto_id=True)

  # Max frame number maybe different for gt and t files
  for frame in range(int(gt[:,0].max())):
    frame += 1 # detection and frame numbers begin at 1

    # select id, x, y, width, height for current frame
    # required format for distance calculation is X, Y, Width, Height \
    # We already have this format
    gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
    t_dets = t[t[:,0]==frame,1:6] # select all detections in t

    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=0.5) # format: gt, t

    # Call update once for per frame.
    # format: gt object ids, t object ids, distance
    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)
    
  return acc

def one_track(trk):
    names = []
    accs = []

    for seq_id, seq in enumerate(dataset):
        base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
        gt = []
        pred = []
        for (obj_id, obj_gt) in list(seq.ground_truth_rect.items()):
            results_path = '{}_{}.txt'.format(base_results_path,obj_id)
            if os.path.isfile(results_path):
                pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
                obj_gt = torch.tensor(obj_gt,dtype=torch.float64)
            else:
                print('Result not found. {}'.format(results_path))
                continue
            for frame_id, b in enumerate(obj_gt):
                if b[0] == -1: continue
                gt.append([frame_id+1,obj_id,b[0],b[1],b[2],b[3],1.,-1.,-1.,-1.])
            for frame_id, b in enumerate(pred_bb):
                if b[0] == -1: continue
                pred.append([frame_id+1,obj_id,b[0],b[1],b[2],b[3],1.,-1.,-1.,-1.])
        # if seq.name in ['0043f083b5','0044fa5fba']:
        #     slices = slice(2,6)
        #     print(f'----------------{seq.name}-------------')
        #     cur_id = 0
        #     for i,j in zip(gt,pred):
        #         if i[1] != cur_id:
        #             print(f'----------------{i[1]}-------------')
        #             cur_id = i[1]
        #         a = [f'{int(ii)-int(jj):>3}' for ii,jj in zip(i[slices],j[slices])]
        #         bi = [int(ii) for ii in i[slices]]
        #         bj = [int(jj) for jj in j[slices]]
        #         c = iou_matrix([i[slices]],[j[slices]])[0][0]
        #         print(f'{c:>5.3f}',a,bi,bj)
        if len(gt) == 0: continue
        accs.append(motMetricsEnhancedCalculator(np.array(gt, dtype=np.float64), np.array(pred, dtype=np.float64)))
        names.append(seq.name)


    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=['mota', 'motp', 'num_frames', 'num_objects' , 'idf1', 'idp', 'idr', \
                                        'recall', 'precision',  \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations'
                                    ], \
                        generate_overall=True,
                        names=names)

    # strsummary = mm.io.render_summary(
    #     summary,
    #     #formatters={'mota' : '{:.2%}'.format},
    #     namemap={'num_objects': 'GT', 'mota': 'MOTA', 'motp' : 'MOTP', \
    #              'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
    #             'precision': 'Prcn',  \
    #             'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
    #             'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
    #             'num_misses': 'FN', 'num_switches' : 'IDsw', \
    #             'num_fragmentations' : 'FM'
    #             }
    # )

    df =  pd.DataFrame.from_dict(summary)
    namemap = {'num_objects': 'GT', 'mota': 'MOTA', 'motp' : 'MOTP', \
                 'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', 'precision': 'Prcn',  \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'num_frames': 'Frames'
                }
    df.rename(columns=namemap, inplace=True)
    return df

if not os.path.exists('../tracking_csv'):
    os.makedirs('../tracking_csv')
for i in range(len(tkl)):
    if os.path.exists(os.path.join('../tracking_csv', 'result_{}_{}.csv'.format(tkl[i][0], tkl[i][1]))):
        print(f'{tkl[i][0]} {tkl[i][1]} already exists')
        continue
    result = one_track(tkl[i][-1])
    trk,param = tkl[i][0], tkl[i][1]
    result.to_csv(os.path.join('../tracking_csv', 'result_{}_{}.csv'.format(trk, param)), index=False)
    print(f'{tkl[i][0]} {tkl[i][1]} done')