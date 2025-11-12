# -*- coding: utf-8 -*-
import numpy as np
from ChannelModel import global_channel_model
from logger import debug, debug_print
from Parameters import *
from Parameters import (
    V2V_DELAY_THRESHOLD, N_V2I_LINKS, V2I_LINK_POSITIONS, V2I_TX_POWER,
    TRANSMITTDE_POWER, SYSTEM_BANDWIDTH,
    V2V_PACKET_SIZE_BITS, V2V_CHANNEL_BANDWIDTH, V2V_MIN_SNR_DB
)


class NewRewardCalculator:

    def __init__(self):
        self.channel_model = global_channel_model
        self.BEAM_ROLLOFF_EXPONENT = 2
        self.ANGLE_PER_DIRECTION = 10
        debug("NewRewardCalculator initialized (Stable Constraint-based Reward)")

    def _calculate_directional_gain(self, horizontal_dir, vertical_dir):
        theta_h = (horizontal_dir - 1) * self.ANGLE_PER_DIRECTION
        theta_v = (1 - vertical_dir) * self.ANGLE_PER_DIRECTION
        theta_h_rad = np.deg2rad(theta_h)
        theta_v_rad = np.deg2rad(theta_v)
        gain_h = np.cos(theta_h_rad) ** self.BEAM_ROLLOFF_EXPONENT
        gain_v = np.cos(theta_v_rad) ** self.BEAM_ROLLOFF_EXPONENT
        effective_gain = gain_h * gain_v
        return effective_gain

    def _record_communication_metrics(self, dqn, delay, snr):
        """安全记录通信指标到DQN，并增加V2V成功率追踪"""
        try:
            if delay is not None and not np.isnan(delay) and delay > 0:
                dqn.delay_list.append(delay)
            else:
                dqn.delay_list.append(1.0)  # 1.0s 作为惩罚值

            if (snr is not None and not np.isnan(snr) and
                    not np.isinf(snr) and snr > -100):
                dqn.snr_list.append(snr)
            else:
                dqn.snr_list.append(0.0)

            if not hasattr(dqn, 'v2v_success_list'):
                dqn.v2v_success_list = []
            if not hasattr(dqn, 'v2v_delay_ok_list'):
                dqn.v2v_delay_ok_list = []
            if not hasattr(dqn, 'v2v_snr_ok_list'):
                dqn.v2v_snr_ok_list = []

            # 检查两个因子是否都满足
            delay_ok = (delay is not None and not np.isnan(delay) and delay <= V2V_DELAY_THRESHOLD)
            snr_ok = (snr is not None and not np.isnan(snr) and snr >= V2V_MIN_SNR_DB)

            if delay_ok and snr_ok:
                dqn.v2v_success_list.append(1)  # 成功
            else:
                dqn.v2v_success_list.append(0)  # 失败

            # 记录诊断信息
            dqn.v2v_delay_ok_list.append(1 if delay_ok else 0)
            dqn.v2v_snr_ok_list.append(1 if snr_ok else 0)

        except Exception as e:
            debug(f"Error recording metrics for DQN {dqn.dqn_id}: {e}")
            if hasattr(dqn, 'v2v_success_list'):
                dqn.v2v_success_list.append(0)
            dqn.delay_list.append(1.0)
            dqn.snr_list.append(0.0)

    def calculate_complete_reward(self, dqn, vehicles, action):
        debug(f"=== Reward Calculation Start for DQN {dqn.dqn_id} ===")
        debug(f"Action: {action}")

        if not vehicles:
            debug("No vehicles - returning default reward 0.0")
            self._record_communication_metrics(dqn, 1.0, 0.0)
            return 0.0  # [!! 修改 !!] 没车不应该给奖励，返回 0.0

        reward = 0.0
        delay = 1.0
        snr_curr = 0.0


        # 我们希望智能体最大化 "V2V性能余量"，同时最小化 "干扰" 和 "功率"

        W_DELAY = 20 # 延迟权重：
        W_SNR = 0.005  # SNR权重

        W_V2I_INTERFERENCE = 1.0  # V2I 干扰惩罚 (保持高位)
        W_POWER_SAVE = 0.05  # 功率节省惩罚 (保持)


        try:
            # 1. 获取最近的车辆信息
            closest_vehicle = vehicles[0]
            distance_3d = self.channel_model.calculate_3d_distance(
                (dqn.bs_loc[0], dqn.bs_loc[1]), closest_vehicle.curr_loc)

            # 2. 解析动作参数
            beam_count = action[0] + 1
            horizontal_dir = action[1]
            vertical_dir = action[2]
            power_ratio = (action[3] + 1) / 10.0

            # 3. 计算方向性增益
            directional_gain = self._calculate_directional_gain(horizontal_dir, vertical_dir)

            # 4. 计算总发射功率
            total_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain

            # 5. 计算V2V的SNR和延迟
            snr_db, snr_linear, _ = self.channel_model.calculate_snr(
                total_power, distance_3d, bandwidth=V2V_CHANNEL_BANDWIDTH)
            snr_curr = snr_db

            delay = self.calculate_delay(distance_3d, action, directional_gain, snr_linear)

            # 奖励项 1: V2V 延迟奖励 (连续, 但奖励值很小)
            reward += (V2V_DELAY_THRESHOLD - delay) * W_DELAY

            # 奖励项 2: V2V SNR 奖励 (连续, 但奖励值很小)
            reward += (snr_curr - V2V_MIN_SNR_DB) * W_SNR

            # 奖励项 3: V2I 干扰惩罚 (保持不变, 现在是主要惩罚)
            v2i_interference_penalty_linear = 0.0
            agent_tx_pos = (dqn.bs_loc[0], dqn.bs_loc[1])

            for link in V2I_LINK_POSITIONS:
                v2i_rx_pos = link['rx']
                interf_dist = self.channel_model.calculate_3d_distance(agent_tx_pos, v2i_rx_pos)
                pl_db, _, _ = self.channel_model.calculate_path_loss(interf_dist)
                pl_linear = 10 ** (-pl_db / 10)
                v2i_interference_penalty_linear += total_power * pl_linear

            reward -= W_V2I_INTERFERENCE * v2i_interference_penalty_linear

            # 奖励项 4: 功率节省惩罚 (保持不变)
            reward -= W_POWER_SAVE * power_ratio

            debug(
                f"Reward Calc: V2V_Delay_R={((V2V_DELAY_THRESHOLD - delay) * W_DELAY):.2f} | "
                f"V2V_SNR_R={((snr_curr - V2V_MIN_SNR_DB) * W_SNR):.2f} | "
                f"V2I_Penalty={W_V2I_INTERFERENCE * v2i_interference_penalty_linear:.3f} | "
                f"Power_Penalty={W_POWER_SAVE * power_ratio:.3f} | Total={reward:.3f}")

            # 更新SNR历史
            dqn.prev_snr = snr_curr

        except Exception as e:
            debug(f"Error in new reward calculation: {e}")
            reward = -2.0  # 出错则给予固定的大惩罚

        # 确保记录通信指标
        self._record_communication_metrics(dqn, delay, snr_curr)

        debug(f"=== Reward Calculation Complete for DQN {dqn.dqn_id} ===")

        return reward

    def calculate_delay(self, distance_3d, dqn_action, directional_gain=1.0, snr_linear=None):
        """
        计算V2V传输时延 (传输时延 + 传播时延)
        """
        try:
            propagation_delay = distance_3d / 3e8

            if snr_linear is None:
                beam_count = dqn_action[0] + 1
                power_ratio = (dqn_action[3] + 1) / 10.0
                tx_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain
                _, snr_linear, _ = self.channel_model.calculate_snr(
                    tx_power, distance_3d, bandwidth=V2V_CHANNEL_BANDWIDTH)

            if snr_linear > 0:
                data_rate = V2V_CHANNEL_BANDWIDTH * np.log2(1 + snr_linear)
                if data_rate > 1e-9:  # 避免除零
                    transmission_delay = V2V_PACKET_SIZE_BITS / data_rate
                else:
                    transmission_delay = 1.0  # 无法传输
            else:
                transmission_delay = 1.0  # 无法传输 (SNR为负)

            delay = transmission_delay + propagation_delay

        except Exception as e:
            debug(f"Error in new delay calculation: {e}")
            delay = 1.0

        return delay

    def calculate_snr_with_direction(self, tx_power, distance_3d, horizontal_dir, vertical_dir):
        directional_gain = self._calculate_directional_gain(horizontal_dir, vertical_dir)
        effective_tx_power = tx_power * directional_gain
        return self.channel_model.calculate_snr(effective_tx_power, distance_3d)

    def get_csi_for_state(self, vehicle, dqn):
        if vehicle is None:
            return [0.0] * 5

        try:
            distance_3d = self.channel_model.calculate_3d_distance(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)

            csi_info = self.channel_model.get_channel_state_info(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc, TRANSMITTDE_POWER,
                bandwidth=V2V_CHANNEL_BANDWIDTH)

            csi_state = [
                csi_info['distance_3d'],
                csi_info['path_loss_total_db'],
                csi_info['shadowing_db'],
                csi_info['snr_db'],
                dqn.prev_snr
            ]

        except Exception as e:
            debug(f"Error getting CSI state: {e}")
            csi_state = [0.0] * 5

        return csi_state


# 全局实例
new_reward_calculator = NewRewardCalculator()