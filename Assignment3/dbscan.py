import os
import sys
import json
import copy
import math
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

tmp_cluster = -1
data_points_dict = {}
neighborhood_dict = {} # {key1 : [], key2 : []}
result_dict = {} # {cluser 1 : [], cluster 2 : [] , ... }

def find_nearby_points(epsilon):
    global tmp_cluster, result_dict, neighborhood_dict, data_points_dict

    points = [(v['x'], v['y']) for v in data_points_dict.values()]
    ids = list(data_points_dict.keys())

    # Create KDTree from points
    tree = KDTree(points)

    # Dictionary to store neighbors
    neighborhood_dict = {id: [] for id in ids}
        # Find all points within epsilon distance for each point
    for i, point in enumerate(points):
        # Query the tree for all points within epsilon distance
        indices = tree.query_ball_point(point, epsilon)
        
        # Translate indices back to original IDs and exclude self from neighbors
        neighborhood_dict[ids[i]] = sorted([ids[j] for j in indices])

def main(train_fname, n, eps, min_pts):
    global tmp_cluster, result_dict, neighborhood_dict, data_points_dict
    # txt 파일읽기
    input_df = pd.read_csv(train_fname,sep='\t',index_col=0,names=['x','y'])

    # 파일 읽어서 x,y 좌표 label 값 초기화 
    for idx,row in input_df.iterrows():
        data_points_dict[idx] = {
            'x' : float(row['x']), 'y': float(row['y']), 'label' : None
        }

    # 각 idx 별로 epsilon 이내인 값 가져오기 
    find_nearby_points(eps)

    for idx, point in data_points_dict.items():
        # 1. 만일 label 있으면 continue
        if point['label'] is not None: continue
        
        # 2. 이웃 Neightbor N_list 들 도출 
        #   ● 만일 Neightbor 갯수들이 minPts 보다 작을시 label(p) = Noise 로 설정이후 continue        
        n_list = neighborhood_dict[idx]
        if len(n_list) < min_pts:
            data_points_dict[idx]['label'] = 'noise'
            continue
        
        # 3. Neighbor 갯수가 minPts 보다 같거나 클시 
        #  ● 들어가야할 cluster label 잡는다. 이거는 Core 이기 때문
        #  ● 현재 내 점인 P를 제외한 N 집합 즉, N-p 인 S집합을 잡는다. 
        tmp_cluster += 1
        data_points_dict[idx]['label'] = tmp_cluster
        
        # 4. for q in S집합 돌면서 이웃인애들 돌면서 
        for q in n_list:
            if q == idx: continue
        
            #  ● label(q) noise 일시 Cluser 로 편입시킨다.
            if data_points_dict[q]['label'] == 'noise':
                data_points_dict[q]['label'] = tmp_cluster

            # ● label 값이 defined 되어있을시 continue 시킨다.abs
            if data_points_dict[q]['label'] is not None: continue

            #  ● q의 neighbor 값들을 구한다. label(q) 를 C 로 편입시킨다.
            q_neighbors = neighborhood_dict[q]
            data_points_dict[q]['label'] = tmp_cluster

            #  ● 만일 q의 len(q)가 minPTs 보다 작을시 컨너뛴다. 걍 
            if len(q_neighbors) < min_pts: continue
            
            #  ● 근데 core 값이면 요놈의 이웃들 전체 중복되지는 않게 추가시켜준다
            for q_neighbor in q_neighbors:
                if q_neighbor not in n_list:
                    n_list.append(q_neighbor)

    # 결과값 result dict 에저장
    for idx, point in data_points_dict.items():
        if isinstance(point['label'], int) is True:
            label = point['label']
            if label in result_dict: result_dict[label].append(idx)
            else: result_dict[label] = [idx]

    # 결과값을 cluster의 크기만큼 잘라야한다. 이떄 ascending 형식으로 해서 큰거
    result_dict = dict(sorted(result_dict.items(), key=lambda item: len(item[1]), reverse=True)[:n])

    # 결과값 
    tmp_cluster = 0
    input_name = os.path.basename(train_fname).split('.')[0]
    for key, values in result_dict.items():
        output_f_name = f'{input_name}_cluster_{tmp_cluster}.txt'
        # 해당 파일에 object_id를 저장합니다.
        with open(output_f_name, 'w') as file:
            for object_id in values:
                file.write(f'{object_id}\n')
        tmp_cluster+=1

if __name__ == '__main__':
    input_data_file = str(sys.argv[1])
    n = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    min_pts = int(sys.argv[4])
    main(input_data_file, n , epsilon, min_pts)