# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from logger import debug, debug_print
from Parameters import *


class EnhancedHeteroGNN(nn.Module):
    """
    异构图神经网络模型
    处理包含RSU和车辆节点的动态图，使用图注意力机制
    """

    def __init__(self, node_feature_dim=7, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super(EnhancedHeteroGNN, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        from GraphBuilder import global_graph_builder
        comm_edge_dim = global_graph_builder.comm_edge_feature_dim
        self.edge_feature_dim = comm_edge_dim

        # 边类型
        self.edge_types = ['communication', 'interference', 'proximity']

        # 节点类型编码（RSU: 0, Vehicle: 1）
        self.node_type_embedding = nn.Embedding(2, hidden_dim // 4)

        # 为每种边类型创建独立的GAT层
        self.edge_type_layers = nn.ModuleDict()

        for edge_type in self.edge_types:
            # 第一层：输入维度 = 节点特征 + 节点类型嵌入
            input_dim = node_feature_dim + (hidden_dim // 4)
            edge_layers = nn.ModuleList()
            # 第一层
            edge_layers.append(
                GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout,
                        edge_dim=self.edge_feature_dim)  # <--- 添加 edge_dim
            )
            # 中间层
            for _ in range(num_layers - 2):
                edge_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout,
                            edge_dim=self.edge_feature_dim)  # <--- 添加 edge_dim
                )
            # 最后一层
            if num_layers > 1:
                edge_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout,
                            edge_dim=self.edge_feature_dim)  # <--- 添加 edge_dim
                )

            self.edge_type_layers[edge_type] = edge_layers

        # 边类型注意力权重（学习不同边类型的重要性）
        self.edge_type_attention = nn.Parameter(torch.ones(len(self.edge_types)))
        # 为车辆聚合定义注意力层
        self.attn_pool_linear = nn.Linear(hidden_dim, 1)
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接全局和局部特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, RL_N_ACTIONS)  # 输出Q值
        )

        # 初始化参数
        self._init_weights()
        debug(f"EnhancedHeteroGNN with multi-head attention initialized")

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, graph_data, dqn_id=None):
        """
        前向传播
        """
        node_features = graph_data['node_features']['features']
        node_types = graph_data['node_features']['types']
        edge_features = graph_data['edge_features']

        batch_size = node_features.size(0)

        # 1. 节点类型嵌入
        type_embedding = self.node_type_embedding(node_types)  # [num_nodes, hidden_dim//4]

        # 2. 拼接节点特征和类型嵌入
        x = torch.cat([node_features, type_embedding], dim=1)  # [num_nodes, node_feature_dim + hidden_dim//4]

        # 3. 为每种边类型分别进行图卷积
        edge_outputs = []
        edge_weights = F.softmax(self.edge_type_attention, dim=0)

        for i, edge_type in enumerate(self.edge_types):
            if edge_features[edge_type] is None:
                edge_outputs.append(torch.zeros(batch_size, self.hidden_dim, device=x.device))
                continue

            edge_index = edge_features[edge_type]['edge_index']
            edge_attr = edge_features[edge_type]['edge_attr']

            layers = self.edge_type_layers[edge_type]
            x_edge = x.clone()

            for j, layer in enumerate(layers):
                x_edge = layer(x_edge, edge_index, edge_attr=edge_attr)
                if j < len(layers) - 1:
                    x_edge = F.elu(x_edge)
                    x_edge = F.dropout(x_edge, p=self.dropout, training=self.training)

            x_edge = x_edge * edge_weights[i]
            edge_outputs.append(x_edge)

        # 4. 合并不同边类型的输出（加权平均）
        if len(edge_outputs) > 0:
            # 堆叠所有边类型的输出 [num_edge_types, num_nodes, hidden_dim]
            stacked_outputs = torch.stack(edge_outputs, dim=0)
            # 加权平均 [num_nodes, hidden_dim]
            x_combined = torch.sum(stacked_outputs, dim=0)
        else:
            x_combined = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 5. 提取特征
        if dqn_id is not None:
            # 提取特定DQN的局部特征
            q_values = self._extract_local_features(x_combined, graph_data, dqn_id)
        else:
            # 全局特征聚合
            q_values = self._extract_global_features(x_combined, graph_data)

        return q_values

    def _extract_local_features(self, node_embeddings, graph_data, dqn_id):
        """
        提取特定DQN的局部特征
        """
        nodes = graph_data['nodes']

        # 找到目标RSU节点
        target_rsu_index = -1
        for i, rsu_node in enumerate(nodes['rsu_nodes']):
            if rsu_node['original_id'] == dqn_id:
                target_rsu_index = i
                break

        if target_rsu_index == -1:
            debug(f"Warning: DQN {dqn_id} not found in graph")
            return torch.zeros(RL_N_ACTIONS, device=node_embeddings.device)

        rsu_embedding = node_embeddings[target_rsu_index]

        vehicle_embeddings = []  # 存储车辆嵌入
        vehicle_indices = []  # 存储车辆的节点索引

        for vehicle_node in nodes['vehicle_nodes']:
            for edge in graph_data['edges']['communication']:
                if (edge['source'] == f"rsu_{dqn_id}" and
                        edge['target'] == vehicle_node['id']):
                    vehicle_index = len(nodes['rsu_nodes']) + nodes['vehicle_nodes'].index(vehicle_node)
                    vehicle_embeddings.append(node_embeddings[vehicle_index])
                    break

        if vehicle_embeddings:
            # 将列表转换为张量 [num_vehicles, hidden_dim]
            vehicle_stack = torch.stack(vehicle_embeddings)

            # 1. 计算注意力分数 (简单版本：只基于车辆自身特征)
            attn_scores = self.attn_pool_linear(vehicle_stack)
            # 2. 转换为权重 (softmax)
            attn_weights = F.softmax(attn_scores, dim=0)

            # 3. 计算加权和
            vehicle_embedding = torch.mm(attn_weights.t(), vehicle_stack).squeeze(0)

        else:
            # 如果没有车辆，使用零向量
            vehicle_embedding = torch.zeros_like(rsu_embedding)

        # 拼接RSU和车辆特征
        combined_features = torch.cat([rsu_embedding, vehicle_embedding], dim=0)  # [hidden_dim * 2]

        # 通过输出层得到Q值
        q_values = self.output_layer(combined_features)  # [num_actions]

        return q_values

    def _extract_global_features(self, node_embeddings, graph_data):
        """
        提取全局特征（所有DQN的聚合）

        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            graph_data: 图数据

        Returns:
            q_values: 全局Q值 [num_dqns, num_actions]
        """
        nodes = graph_data['nodes']
        num_rsus = len(nodes['rsu_nodes'])

        all_q_values = []

        # 为每个DQN计算Q值
        for dqn_id in range(1, num_rsus + 1):
            q_value = self._extract_local_features(node_embeddings, graph_data, dqn_id)
            all_q_values.append(q_value)

        # 堆叠所有DQN的Q值 [num_dqns, num_actions]
        if all_q_values:
            return torch.stack(all_q_values, dim=0)
        else:
            return torch.zeros(0, RL_N_ACTIONS, device=node_embeddings.device)

    def get_attention_weights(self, graph_data):
        """
        获取注意力权重（用于分析）

        Returns:
            attention_info: 注意力权重信息
        """
        attention_info = {
            'edge_type_weights': F.softmax(self.edge_type_attention, dim=0).detach().cpu().numpy(),
            'edge_types': self.edge_types
        }

        return attention_info


# 全局GNN模型实例
global_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=8,  # 与GraphBuilder输出一致
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)

# 全局GNN目标网络实例 (结构与在线网络完全相同)
global_target_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=8,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)

def update_target_gnn():
    """
    将在线 GNN 的权重复制到目标 GNN。
    """
    global_target_gnn_model.load_state_dict(global_gnn_model.state_dict())
    global_target_gnn_model.eval() # 确保目标网络处于评估模式
    debug("Global Target GNN network updated")

def update_target_gnn_soft(tau):
    """
    软更新目标 GNN 网络。
    target_param = tau * online_param + (1 - tau) * target_param
    """
    try:
        with torch.no_grad():
            for target_param, online_param in zip(global_target_gnn_model.parameters(), global_gnn_model.parameters()):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
        # debug("Global Target GNN network updated (Soft)") # (太频繁, 注释掉)
    except Exception as e:
        debug(f"Error during GNN soft update: {e}")

# 初始化: 确保目标网络权重与在线网络一致
update_target_gnn()
debug_print("Global GNN 和 Target GNN 已初始化并同步。")

def test_gnn_model():
    """测试GNN模型"""
    from GraphBuilder import global_graph_builder
    from Classes import DQN, Vehicle
    import torch

    debug_print("Testing GNN Model...")

    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dqn_list = []
    test_vehicle_list = []

    # 创建测试DQN
    test_dqn = DQN(
        RL_N_STATES, RL_N_HIDDEN, RL_N_ACTIONS,
        dqn_id=1, start_x=0, start_y=400, end_x=400, end_y=400
    )
    test_dqn.vehicle_exist_curr = True
    test_dqn.prev_snr = 25.0
    test_dqn.vehicle_in_dqn_range_by_distance = []
    test_dqn_list.append(test_dqn)

    # 创建测试车辆
    test_vehicle = Vehicle(1, 100, 400, 1, 0)
    test_vehicle_list.append(test_vehicle)

    test_vehicle2 = Vehicle(2, 300, 400, -1, 0)
    test_vehicle_list.append(test_vehicle2)

    # 构建图
    graph_data = global_graph_builder.build_dynamic_graph(
        test_dqn_list, test_vehicle_list, epoch=1)

    # 将数据移动到设备
    graph_data['node_features']['features'] = graph_data['node_features']['features'].to(device)
    graph_data['node_features']['types'] = graph_data['node_features']['types'].to(device)

    for edge_type in global_gnn_model.edge_types:
        if graph_data['edge_features'][edge_type] is not None:
            graph_data['edge_features'][edge_type]['edge_index'] = \
                graph_data['edge_features'][edge_type]['edge_index'].to(device)
            graph_data['edge_features'][edge_type]['edge_weights'] = \
                graph_data['edge_features'][edge_type]['edge_weights'].to(device)

    # 将GNN模型移动到设备
    gnn_model = global_gnn_model.to(device)

    # 测试前向传播
    with torch.no_grad():
        # 测试全局特征
        global_q_values = gnn_model(graph_data)
        debug(f"Global Q values shape: {global_q_values.shape}")

        # 测试局部特征
        local_q_values = gnn_model(graph_data, dqn_id=1)
        debug(f"Local Q values for DQN 1: {local_q_values.shape}")

        # 测试注意力权重
        attention_info = gnn_model.get_attention_weights(graph_data)
        debug(f"Edge type attention weights: {attention_info}")

    # 测试GNNDQN包装器
    gnn_dqn = GNNDQN(gnn_model, test_dqn)
    gnn_dqn_q_values = gnn_dqn(graph_data)
    debug(f"GNNDQN Q values: {gnn_dqn_q_values.shape}")

    debug_print("GNN Model test completed!")


if __name__ == "__main__":
    set_debug_mode(True)
    test_gnn_model()