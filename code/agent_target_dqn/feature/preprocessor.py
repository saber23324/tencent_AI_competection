#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
import random
from agent_target_dqn.feature.definition import reward_process
from agent_target_dqn.conf.myconfig import Args

def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.reset()

    def reset(self):
        self.target_pos_list = [(26, 87), (85, 114), (32, 24), (101, 40), (59, 64)]
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        ## 自定义
        self.end_pos_obs = (0,0)
        self.map_walk = np.zeros((128, 128))
        self.hitwall_flag = 0
        self.terminated_flag = False
        self.treasure_score = 0
        self.organ_pos = [(0, 0) for _ in range(23)] # 物件id 0代表buff，1~13代表宝箱 21代表起点, 22代表终点
        self.organ_dist = np.zeros(23, dtype=float)
        self.organ_dist_last =np.zeros(23, dtype=float)
        self.miss_treasure = 0.0
        self.last_dist_end = 0.0
        self.dist_end = 0
        self.is_treasure_found = np.zeros(23, dtype=bool)
        self.memory_map = np.zeros((51, 51))
        self.obstacle_map = np.zeros((51, 51))
        self.treasure_map = np.zeros((51, 51))
        self.end_map = np.zeros((51, 51))

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, ex_obs = frame_state[0],frame_state[1]
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        if self.end_pos is None:
            self.end_pos = self.target_pos_list.pop(random.randrange(len(self.target_pos_list)))

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)
        # 更新地图：
        if self.map_walk[hero["pos"]["x"]][hero["pos"]["z"]] < 1.0:
            self.map_walk[hero["pos"]["x"]][hero["pos"]["z"]]  +=  0.1
        else:
            self.map_walk[hero["pos"]["x"]][hero["pos"]["z"]]  =  1.0
# 提取memory地图
        center_x, center_z = self.cur_pos[0], self.cur_pos[1]  # 当前点坐标
        # 创建51x51的零矩阵
        self.memory_map = np.zeros((51, 51), dtype=self.map_walk.dtype)

        # 计算在完整地图中的有效范围
        src_x_start = max(0, center_x - 25)
        src_x_end = min(128, center_x + 26)
        src_z_start = max(0, center_z - 25)
        src_z_end = min(128, center_z + 26)

        # 计算在目标矩阵中的放置位置
        dst_x_start = max(0, 25 - center_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_z_start = max(0, 25 - center_z)
        dst_z_end = dst_z_start + (src_z_end - src_z_start)

        # 将有效区域复制到目标矩阵中
        self.memory_map[dst_x_start:dst_x_end, dst_z_start:dst_z_end] = \
            self.map_walk[src_x_start:src_x_end, src_z_start:src_z_end]
        for i,dir in enumerate(obs["map_info"]):
            self.obstacle_map[i] = dir["values"]



        # 宝箱奖励
        # self.treasure_score = ex_obs["game_info"]["treasure_score"]



        # self.miss_treasure = ex_obs["game_info"]["treasure_count"] - ex_obs["game_info"]["treasure_collected_count"]
        
         # End position
        # 终点位置
        self.organ_dist_last =np.copy(self.organ_dist)

        #更新 先清零
        self.treasure_map = np.zeros((51, 51))
        self.end_map = np.zeros((51, 51))
        
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4 and organ["status"] != -1:
                config_id = organ["config_id"]
                self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                pos_x = organ["pos"]["x"] - self.cur_pos[0] + 25
                pos_z = organ["pos"]["z"] - self.cur_pos[1] + 25
                self.end_map[pos_x][pos_z] = 1
                # 计算该物体与当前位置的距离
                self.organ_pos[config_id] = (organ["pos"]["x"], organ["pos"]["z"])
                relative_pos = tuple(y - x for x, y in zip(self.cur_pos, self.organ_pos[config_id]))
                self.organ_dist[config_id] = np.linalg.norm(relative_pos)
                self.is_end_pos_found = True
                

            if organ["sub_type"] == 1:
                config_id = organ["config_id"]
                pos_x = organ["pos"]["x"] - self.cur_pos[0] + 25
                pos_z = organ["pos"]["z"] - self.cur_pos[1] + 25
                if organ["status"] == 1:
                    self.is_treasure_found[config_id] = True
                    self.treasure_map[pos_x][pos_z] = 1
                    self.organ_pos[config_id] = (organ["pos"]["x"], organ["pos"]["z"])
                    # 计算该物体与当前位置的距离
                    relative_pos = tuple(y - x for x, y in zip(self.cur_pos, self.organ_pos[config_id]))
                    self.organ_dist[config_id] = np.linalg.norm(relative_pos)
                else:
                    self.treasure_map[pos_x][pos_z] = -1
                    self.is_treasure_found[config_id] = False
                    self.organ_dist[config_id] = -1#NotFound
            if organ["sub_type"] == 2:#buff
                config_id = organ["config_id"]
                pos_x = organ["pos"]["x"] - self.cur_pos[0] + 25
                pos_z = organ["pos"]["z"] - self.cur_pos[1] + 25
                if organ["status"] == 1: 
                    self.is_treasure_found[config_id] = True
                    self.treasure_map[pos_x][pos_z] = 2
                    self.organ_pos[config_id] = (organ["pos"]["x"], organ["pos"]["z"])
                    # 计算该物体与当前位置的距离
                    relative_pos = tuple(y - x for x, y in zip(self.cur_pos, self.organ_pos[config_id]))
                    self.organ_dist[config_id] = np.linalg.norm(relative_pos)
                else:
                    self.treasure_map[pos_x][pos_z] = -2
                    self.is_treasure_found[config_id] = False
                    self.organ_dist[config_id] = -1#NotFound



            self.is_end_pos_found = False
        target_relative_pos = tuple(y - x for x, y in zip(self.cur_pos, self.end_pos))
        target_dist = np.linalg.norm(target_relative_pos)

        # if end_pos is not found, try to change to a new random target
        # 如果终点位置未找到，尝试更换随机的新目标
        if not self.is_end_pos_found:
            if target_dist < 10 and len(self.target_pos_list) > 0:
                self.end_pos = self.target_pos_list.pop(random.randrange(len(self.target_pos_list)))

        self.last_dist_end = self.dist_end
        # # self.end_pos_obs = (ex_obs["game_info"]["end_pos"]["x"], ex_obs["game_info"]["end_pos"]["z"])#传进来不能直接用？？
        # relative_pos = tuple(y - x for x, y in zip(self.cur_pos, self.end_pos))
        self.dist_end = target_dist
       

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action
    def update_terminated(self,terminated):
        self.terminated_flag = terminated[0]
        self.end_pos_obs = terminated[1]
    def reward_pross(self):
        # REWARD
        # 1. punish repeated step around
        ratio =self.step_no/Args['max_env_step'] # 计算步数占总步数的比例
        # 提取中间5x5的矩阵
        center_x, center_z = self.cur_pos[0],self.cur_pos[1]  # 当前点坐标
        sub_matrix = self.map_walk[center_x-2:center_x+3, center_z-2:center_z+3]
        # 拉直为一维数组
        obs_data_5_5 = sub_matrix.flatten()
        if ratio<0.5:
            around_reward = -max(self.map_walk[center_x][center_z] -  Args['repeat_step_thre'], 0)
        else:
            around_reward = -(Args['repeat_punish5_5'] * np.maximum(
                obs_data_5_5-Args['repeat_step_thre'], 0).reshape(5,5)).sum()
        # obs_data_10_10 = sub_matrix.flatten()
        # if ratio<0.5:
        #     around_reward = -max(self.map_walk[center_x][center_z] - 0.1 * Args['repeat_step_thre'], 0)
        # else:
        #     around_reward = -(Args['repeat_punish'] * np.maximum(
        #         obs_data_10_10-0.1*Args['repeat_step_thre'], 0).reshape(11,11)).sum()
        # 2.到终点奖励
        # 稠密奖励
        if self.last_dist_end != 0:
            dist_reward = (self.last_dist_end-self.dist_end)*Args['dist_reward_coef']
        else :
            dist_reward = 0
        if self.terminated_flag:
            final_reward = 150
            # punish treasures haven't get
            # final_reward = final_reward - self.miss_treasure * Args['treasure_punish_coef']
        else :
            final_reward = 0
        
        # 3.hit wall
        if self.hitwall_flag >= 3:
            hitwall_reward = -3
        else:
            hitwall_reward = 0
        # 4. treasure

        # 只考虑有效的宝箱ID范围(1-13)
        valid_indices = slice(1, 14)
        treasure_found_mask = self.is_treasure_found[valid_indices]
        if np.any(treasure_found_mask):
            dist_improvement = (self.organ_dist_last[valid_indices] - self.organ_dist[valid_indices]) * treasure_found_mask
            treasure_reward = np.max(dist_improvement)*Args['dist_reward_coef']*1.2 #额外增益
        else:
            treasure_reward = 0
        # if not self.terminated_flag and self.treasure_score == 100:
        #     r += 50 * (self.treasure_reward_coef * (self._last_treasure_flag - obs_data[239:249])).sum()
        ant_dist_treasure = np.max(self.organ_dist[1:14])
        ter_reward = ant_dist_treasure*Args['dist_reward_coef']
        # step reward
        # 步数奖励
        step_reward = -0.01
        # reward = (around_reward+ dist_reward + hitwall_reward + ter_reward)/Args['rate_of_projection']
        reward = ( dist_reward + hitwall_reward + final_reward + around_reward + step_reward)/Args['rate_of_projection']
        return [reward,around_reward ,dist_reward , treasure_reward , final_reward ,self.map_walk[center_x][center_z]]
        

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        # 添加视野特征
        classical_feature = np.concatenate([
        # 官方
        self.cur_pos_norm,
        self.feature_end_pos, 
        self.feature_history_pos, 
        legal_action,
        ])
        feature = np.concatenate([
        np.stack([  # Image: (4, 51, 51)(10404)
        self.obstacle_map,
        self.memory_map,
        self.treasure_map,
        self.end_map,
        ], axis=0).reshape(-1),
        # 自定义特征
        self.organ_dist, #23
        self.is_treasure_found,#23
        # 官方
        classical_feature, # Feature
        ]).astype(np.float32)
        # assert feature.shape[0] == Args[], f"ERROR: {feature.shape[0]=} != {args.obs_dim+1}"

        return (
            feature,
            legal_action,
            self.reward_pross()
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
            self.hitwall_flag += 1 
        else:
            self.bad_move_ids = set()
            self.hitwall_flag = 0

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action
