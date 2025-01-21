---
title: "PNC Utils"
date: "2024-12-14"
author: "lvsolo"
tags: ["PNC", "Nuplan", "planning"]
---

# Contents
- [Contents](#contents)
- [I. Nuplan](#i-nuplan)
  - [1. nuboard UI module create](#1-nuboard-ui-module-create)
  - [2. collect the infos of nuplan dataset](#2-collect-the-infos-of-nuplan-dataset)
    - [1) Collect tokens of some scenario type from db files](#1-collect-tokens-of-some-scenario-type-from-db-files)
  - [3. split the yamls to train](#3-split-the-yamls-to-train)
    - [1) auto split the yaml files](#1-auto-split-the-yaml-files)
    - [2) collect and join the split yamls' metrics](#2-collect-and-join-the-split-yamls-metrics)
    - [3) find the bad cases](#3-find-the-bad-cases)



# I. Nuplan
## 1. nuboard UI module create
   ```python
    #main_callbacks.on_run_simulation_end()
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    #builder = cfg.scenario_builder
    builder = NuPlanScenarioBuilder('/mnt/HDD/dataset/nuplan/', '/mnt/HDD/dataset/nuplan/mini/maps', None, None, 'nuplan-maps-v1.1', scenario_mapping=scenario_mapping)
    output_dir = '/mnt/HDD/dataset/nuplan/mini/exp/exp/simulation/closed_loop_nonreactive_agents/pluto_planner/mini_demo_scenario/'
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']
    nuboard = NuBoard(
        nuboard_paths=simulation_file,
        scenario_builder=builder,#scenario_builder,
        #scenario_builder=runners[0].scenario,#scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )
    nuboard.run()
   ```
## 2. collect the infos of nuplan dataset
   ### 1) Collect tokens of some scenario type from db files
   ```python
from typing import Optional, Union, List
import glob 
from collections import defaultdict as ddict
import sqlite3
from pathlib import Path
from tqdm import tqdm
def get_tokens_from_db(db_path, scenario_type):
    # print(db_path)
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
     
    # 查询数据库中的所有表名
    tabel = 'scenario_tag'
    # cursor.execute(f"SELECT * FROM scenario_tag;")
    cursor.execute(f"SELECT * FROM {tabel} WHERE type is '{scenario_type}';")
    # cursor.execute(f"SELECT * FROM {tabel} WHERE type LIKE '%high%'  AND type LIKE '%lateral%';")
    # cursor.execute(f"SELECT * FROM {tabel} WHERE 'type' = 'high_lateral_acceleration'")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    # 将每一行的数据转换为字典，并处理字节串
    dict_rows = []
    for row in rows:
        new_row = {}
        for col, value in zip(columns, row):
            if isinstance(value, bytes):
                # 如果值是字节串，使用.hex()方法转换为字符串
                new_row[col] = value.hex()
            else:
                new_row[col] = value
        dict_rows.append(new_row)
    return dict_rows

def get_tokens_from_dbs(db_dir: Union[str, List[str]], scenario_type: Union[str, List[str]]):
    scenario_types = []
    db_dirs = []
    if isinstance(db_dir, str):
        db_dirs = [db_dir]
    elif isinstance(db_dir, list):
        db_dirs = db_dir
    if isinstance(scenario_type, str):
        scenario_types = [scenario_type]
    elif isinstance(scenario_type, list):
        scenario_types = scenario_type
    map_scenario_type_to_token = ddict(list)
    for dir in tqdm(db_dirs):
        for db_path in tqdm(Path(dir).glob("*.db")):
            for scenario in scenario_types:
                # print(scenario)
                tmp_tokens = get_tokens_from_db(db_path, scenario)
                # print(tmp_tokens)
                map_scenario_type_to_token[scenario].extend(tmp_tokens)
    return map_scenario_type_to_token

interest_scenario_type=["starting_left_turn","starting_right_turn","following_lane_with_lead",
                        "waiting_for_pedestrian_to_cross","traversing_pickup_dropoff","low_magnitude_speed",
                        "high_lateral_acceleration","near_multiple_vehicles","high_magnitude_speed",
                        "starting_straight_traffic_light_intersection_traversal","changing_lane",
                        "stationary_in_traffic","behind_long_vehicle","stopping_with_lead"]
ret_tokens = get_tokens_from_dbs('nuplan/nuplan-v1.1/data/cache/train/', interest_scenario_type)
val_tokens = get_tokens_from_dbs('nuplan/nuplan-v1.1/data/cache/val/', interest_scenario_type)
for k, v in val_tokens.items():
    print(k, len(v))
import json
with open('dict_14_scenario_type_to_tokens_in_val.json', 'w') as json_file:
    json.dump(val_tokens, json_file)
with open('dict_14_scenario_type_to_tokens_in_val.json', 'r') as json_file:
    my_dict_val = json.load(json_file)
my_dict_val['starting_left_turn'][0]['token']
for k,v in my_dict_val.items():
    my_dict_val[k] = [vv['lidar_pc_token'] for vv in v]
    import json
# with open('dict_14_scenario_type_to_tokens_in_train.json', 'w') as json_file:
#     json.dump(ret_tokens, json_file)
with open('dict_14_scenario_type_to_tokens_in_train.json', 'r') as json_file:
    my_dict = json.load(json_file)

my_dict['starting_left_turn'][0]['token']
for k,v in my_dict.items():
    my_dict[k] = [vv['lidar_pc_token'] for vv in v]
import random
sample_number_per_type = {
    'starting_left_turn': 2.5, 
    'starting_right_turn': 2, 
    'following_lane_with_lead': 1.5, 
    'traversing_pickup_dropoff': 1.5, 
    'low_magnitude_speed': 1.5, 
    'waiting_for_pedestrian_to_cross': 1.5, 
    'high_lateral_acceleration': 1, 
    'high_magnitude_speed': 1, 
    'starting_straight_traffic_light_intersection_traversal': 1, 
    'near_multiple_vehicles': 1, 
    'changing_lane': 1, 
    'stopping_with_lead': 0.5, 
    'stationary_in_traffic': 0.5, 
    'behind_long_vehicle': 0.5
}
sampled_tokens_per_type = ddict(list)
for k, v in sample_number_per_type.items():
    v *= 20000
    if len(my_dict[k]) < v:
        v = len(my_dict[k])
    sampled_tokens_per_type[k] = random.sample(my_dict[k], int(v))
   ```

## 3. split the yamls to train 
   ### 1) auto split the yaml files
```bash


ORIGIN_YAML_FILE=$1
NUM_SPLIT=$2
CKPT_PATH=$3

if [[ "$ORIGIN_YAML_FILE" == *.yaml ]]; then
    echo "文件名以 .yaml 结尾"
else
    echo "文件名不是以 .yaml 结尾"
fi

# echo each line of the $ORIGIN_YAML_FILE using linux shell
FLAG_TOKEN_START=0
FLAG_TOKEN_END=0


PREFIX_NAME_YAML="${ORIGIN_YAML_FILE:0:-5}"
declare -a NEW_FPLIT_YAML_NAMES
# generate yamls with index
for ((IND=1; IND<=NUM_SPLIT; IND++)); do
    VAR="${PREFIX_NAME_YAML}_${IND}.yaml"
    echo "$VAR"  # 打印每个生成的变量名
    NEW_FPLIT_YAML_NAMES+=("$VAR")
    rm $VAR
done
echo "new yaml files:"
echo  ${NEW_FPLIT_YAML_NAMES[@]} |tr -s ' ' '\n' 


IND_WHICH_FILE_TO_WRITE_TOKEN=0
while IFS= read -r line; do
  if [[ $FLAG_TOKEN_START -eq 1 && $FLAG_TOKEN_END -eq 0 && $line != *'- '* ]] ;then
    echo "TOKEN END"
    FLAG_TOKEN_START=0
    FLAG_TOKEN_END=1
    # continue
  fi
  
  if [[ $FLAG_TOKEN_START -eq 0 ]] ;then
    for ((IND=1; $IND<=NUM_SPLIT; IND++)); do
        
        #echo ${NEW_FPLIT_YAML_NAMES[$IND-1]} $line
        echo "$line" >> ${NEW_FPLIT_YAML_NAMES[$IND-1]}
    done
  fi

  if [[ $FLAG_TOKEN_START -eq 1 && $FLAG_TOKEN_END -eq 0 ]] ;then
    echo "$line" >> ${NEW_FPLIT_YAML_NAMES[${IND_WHICH_FILE_TO_WRITE_TOKEN}-1]}
    #echo $IND_WHICH_FILE_TO_WRITE_TOKEN $line
    IND_WHICH_FILE_TO_WRITE_TOKEN=$(( (IND_WHICH_FILE_TO_WRITE_TOKEN + 1) % NUM_SPLIT ))  
  fi
  if [[ $FLAG_TOKEN_START -eq 0 && $FLAG_TOKEN_END -eq 0 && $line == *'scenario_tokens'* ]] ;then
    echo "TOKEN START"
    FLAG_TOKEN_START=1
    FLAG_TOKEN_END=0
    # continue
  fi

done < $ORIGIN_YAML_FILE



#DIR_SAVE_SPLIT_VAL_RESULTS="results/split_val"

start=$(date +%s)
for ((IND=1; IND<=NUM_SPLIT; IND++)); do
    NAME_SPLIT_YAML="${NEW_FPLIT_YAML_NAMES[$IND-1]:0:-5}"
    PURE_YAML_FILE_NAME=`echo $NAME_SPLIT_YAML|rev|cut -d '/' -f 1|rev`
    echo ${PURE_YAML_FILE_NAME}
    #bash split_sim_pluto.sh  ${PURE_YAML_FILE_NAME} & 
    bash split_sim_pluto.sh  ${PURE_YAML_FILE_NAME}  ${CKPT_PATH} 2>&1 |tee logs/log_${PURE_YAML_FILE_NAME} & 
done

end=$(date +%s)
runtime=$((end-start))
echo "Command took $runtime seconds"

```
   ### 2) collect and join the split yamls' metrics
   The metrics saved by Pluto or Planscope is in two folders:one is the metrics of each scenario token and the total mean metrics of the whole scenario tokens in the scenario_filter yaml;the other is the aggregated metrics of the whole scenario tokens in the scenario_filter yaml.
Example Scripts:
```python
# metrics for each scenario token
file0='metrics/ego_progress_along_expert_route.parquet'
file1='metrics/ego_is_making_progress.parquet'

df0=pd.read_parquet(file0)
df1=pd.read_parquet(file1)
df0.keys()
df0['ego_expert_progress_along_route_ratio_stat_value']
df0['scenario_name'].count()

# aggregated metrics
dfs=pd.read_parquet('aggregator_metric/2024.12.26.11.26.15.parquet')
dfs.keys()
final_score=dfs[dfs['scenario']=='final_score']
num_tokens = final_score['num_scenarios'].values[0]
num_tokens
final_score.to_dict(orient="records")[0]
dfs['scenario_type'].values[:109]
```

### 3) find the bad cases
```python
import pandas as pd
import os
import sys
import glob
from collections import defaultdict as ddict
# dir_result_splits='results/split_val/'
dir_result_splits='/cpfs/user/lvshoulu/git/PNC/PlanScope/results/split_val/closed_loop_nonreactive_agents/withrule_true/epoch17_ckpt/'
# dir_result_splits='results/split_val/closed_loop_nonreactive_agents/'
index = 1
origin_yaml_pure_name=""
num_split = 10
if index < len(sys.argv):
    origin_yaml_pure_name = sys.argv[index]
    index += 1

if index < len(sys.argv):
    num_split = int(sys.argv[index])
    index += 1

# num_split = 1

# get metrics summary results start
metric_names = ["corners_in_drivable_area","drivable_area_compliance","driving_direction_compliance","ego_is_comfortable","ego_is_making_progress","ego_progress_along_expert_route","no_ego_at_fault_collisions","speed_limit_compliance","time_to_collision_within_bound"]
#"ego_jerk","ego_lane_change","ego_lat_acceleration","ego_lon_acceleration","ego_lon_jerk","ego_yaw_acceleration","ego_yaw_rate",
val_sum_every_metric = {}
count_every_metric = {}
key_in_metric_df = {'driving_direction_compliance':'driving_direction_compliance_score_stat_value',
                    'ego_progress_along_expert_route':'ego_expert_progress_along_route_ratio_stat_value',
                    }
for mn in metric_names:
    val_sum_every_metric[mn] = 0
    count_every_metric[mn] = 0
value_zero_tokens = ddict(list)
token2logname_type = ddict(dict)

for i in range(1, num_split+1):
    cur_dir = os.path.join(dir_result_splits, origin_yaml_pure_name+"_{}".format(str(i)))
    # print(cur_dir)
    try:
        met_path = os.path.join(cur_dir, 'metrics', metric_names[0]+'.parquet')
        df = pd.read_parquet(met_path)
    except:
        print(cur_dir)
        continue
    for mn in metric_names:
        met_path = os.path.join(cur_dir, 'metrics', mn+'.parquet')
        df = pd.read_parquet(met_path)
        try:
            count_every_metric[mn] +=  df[mn+'_stat_value'].count()
            val_sum_every_metric[mn] += df[mn+'_stat_value'].sum()
            for log_name, scenario_token, scenario_type, value in zip(df['log_name'], df['scenario_name'], df['scenario_type'], df[mn+'_stat_value']):
                if scenario_token not in token2logname_type:
                    token2logname_type[scenario_token]= {"log_name":log_name,  'scenario_type':scenario_type}
                if value < 0.5:
                    value_zero_tokens[mn].append({'log_name':log_name, 'scenario_token':scenario_token, 'scenario_type':scenario_type, 'valude':value})
        except:
            count_every_metric[mn] +=  df[key_in_metric_df[mn]].count()
            val_sum_every_metric[mn] += df[key_in_metric_df[mn]].sum()
            for log_name, scenario_token, scenario_type, value in zip(df['log_name'], df['scenario_name'], df['scenario_type'], df[key_in_metric_df[mn]]):
                if value < 0.5:
                    value_zero_tokens[mn].append({'log_name':log_name, 'scenario_token':scenario_token, 'scenario_type':scenario_type, 'valude':value})
            # print(mn)#, df.keys())



bad_score_dict_token=ddict(int)
for mn in metric_names:
    for i in value_zero_tokens[mn]:
        # print('000000:', i)
        bad_score_dict_token[i['scenario_token']] += 1
sorted_bad_token = sorted(bad_score_dict_token.items(), key=lambda x:x[1], reverse=True)
print('-'*40)
print('bad_score_dict_token:')
for i in sorted_bad_token:
    print(i, token2logname_type[i[0]])
# for mn in metric_names:
#     for i in value_zero_tokens[mn]:
#         print(mn, i)
#for mn in metric_names:
#    print(mn, val_sum_every_metric[mn]/count_every_metric[mn], val_sum_every_metric[mn], count_every_metric[mn])

# get metrics summary results end

# get aggregate metrics results start
val_weight_sum_each_split=ddict(float)
count_total_tokens = 0
score_of_each_token = ddict(float)
score_of_each_scenario_type = ddict(float)
count_of_each_scenario_type = ddict(int)
weight_sum_of_final_score = 0.
for i in range(1, num_split+1):
    cur_dir = os.path.join(dir_result_splits, origin_yaml_pure_name+"_{}".format(str(i)), 'aggregator_metric/')
    parquet_files = glob.glob(os.path.join(cur_dir, '*.parquet'))
    sorted_files = sorted(parquet_files, key=lambda x: os.path.getmtime(x), reverse=True)
    try:
        dfs=pd.read_parquet(sorted_files[0])
    except:
        print(cur_dir)
        continue
    final_score=dfs[dfs['scenario']=='final_score']
    num_tokens = final_score['num_scenarios'].values[0]
    final_score = final_score.to_dict(orient="records")[0]
    weight_sum_of_final_score += final_score['score'] * num_tokens
    for mn in metric_names:
        val = final_score[mn]
        if val is None:
            continue
        weight = num_tokens
        val_weight_sum_each_split[mn]+= val*weight
    count_total_tokens += num_tokens

    # score for each token
    for token, score in zip(dfs['scenario'][:int(num_tokens)],dfs['score'][:int(num_tokens)]):
        score_of_each_token[token] = score
    # score for each scenario type
    count_of_each_scenario_type_in_this_split = ddict(int)    
    for st in dfs['scenario_type'].values[:int(num_tokens)]:
        count_of_each_scenario_type[st] += 1
        count_of_each_scenario_type_in_this_split[st]+=1

    for scenario_type, score in zip(dfs['scenario'][int(num_tokens):-1],dfs['score'][int(num_tokens):-1]):
        score_of_each_scenario_type[scenario_type] += score * count_of_each_scenario_type_in_this_split[scenario_type]
for scenario_type, _ in score_of_each_scenario_type.items():
    score_of_each_scenario_type[scenario_type] /= count_of_each_scenario_type[scenario_type]

print('-'*40)
# print('sorted score of each token:')
# for token, score in sorted(score_of_each_token.items(), key=lambda x:x[1], reverse=False):
    # print(token, score)
print('-'*40)
print('sorted score of each type:')
for scenario_type, score in sorted(score_of_each_scenario_type.items(), key=lambda x:x[1], reverse=False):
    print(scenario_type, score)

#print('-'*40)
#print('bad_score_dict_scenario_type:')
#
#sorted_bad_ratio_type = sorted(bad_score_dict_scenario_type.items(), key=lambda x :x[1]/float(count_of_each_scenario_type[x[0]]), reverse=True)
#for k,v in sorted_bad_ratio_type:
#    print(k, v/float(count_of_each_scenario_type[k]), v, count_of_each_scenario_type[k])

bad_score_dict_scenario_type=ddict(int)
for mn in metric_names:
    for i in value_zero_tokens[mn]:
        bad_score_dict_scenario_type[i['scenario_type']] += 1
sorted_bad_type = sorted(bad_score_dict_scenario_type.items(), key=lambda x:x[1], reverse=True)
print('-'*40)
print('bad_score_dict_scenario_type:')
for k,v in sorted_bad_type:
    print(k, v)

print('-'*40)
print('metrics total:')
for mn in metric_names:
    print(mn, val_weight_sum_each_split[mn]/count_total_tokens)
print('final score', weight_sum_of_final_score/count_total_tokens)
print('total tokens', count_total_tokens)
# get aggregate metrics results end
```
