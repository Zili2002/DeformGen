#!/usr/bin/env python3
"""
基于速度修改的软体形变工具

通过直接修改物理点的速度来实现软体局部形变，与CUDA graph模式完全兼容。

核心功能：
- compute_spatial_weights: 计算空间权重（高斯/均匀/线性衰减）
- apply_velocity_deformation_to_env: 对环境施加基于速度的形变

使用示例：
    import gymnasium as gym
    from sim.utils.velocity_deformation_utils import apply_velocity_deformation_to_env

    env = gym.make('BaseEnv-v0', cfg=cfg, ...)
    obs, _ = env.reset(seed=0)

    # 施加0.5 m/s的向上速度
    info = apply_velocity_deformation_to_env(
        env.unwrapped,
        center=[0.31, -0.06, 0.05],
        radius=0.05,
        velocity_magnitude=0.5,
        velocity_direction=[0, 0, 1],
        falloff_type='gaussian'
    )
"""

import torch
import numpy as np
from typing import Union, Dict, Optional


def compute_spatial_weights(
    points: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    falloff_type: str = 'gaussian',
    falloff_sigma: Optional[float] = None
) -> torch.Tensor:
    """
    计算空间权重用于速度修改

    参数:
        points: (N, 3) 所有物理点的位置
        center: (3,) 形变中心的世界坐标
        radius: 影响区域半径（米）
        falloff_type: 衰减类型 - 'uniform', 'gaussian', 或 'linear'
        falloff_sigma: 高斯衰减的标准差（仅用于gaussian类型）

    返回:
        weights: (N,) 张量，值在 [0, 1] 范围内
    """
    # 计算距离
    distances = torch.norm(points - center, dim=1)
    weights = torch.zeros_like(distances)

    # 找到影响范围内的点
    mask = distances < radius

    # 根据衰减类型计算权重
    if falloff_type == 'uniform':
        # 均匀分布：范围内权重为1
        weights[mask] = 1.0

    elif falloff_type == 'gaussian':
        # 高斯分布：平滑衰减
        sigma = falloff_sigma if falloff_sigma else radius / 3.0
        weights[mask] = torch.exp(-distances[mask]**2 / (2 * sigma**2))

    elif falloff_type == 'linear':
        # 线性衰减
        weights[mask] = 1.0 - distances[mask] / radius

    else:
        raise ValueError(f"Unknown falloff_type: {falloff_type}. "
                         f"Must be 'uniform', 'gaussian', or 'linear'")

    return weights


def apply_velocity_deformation_to_env(
    env,
    center: Union[np.ndarray, list],
    radius: float,
    velocity_magnitude: float,
    velocity_direction: Union[np.ndarray, list],
    falloff_type: str = 'gaussian'
) -> Dict:
    """
    通过修改速度实现局部形变

    参数:
        env: Gymnasium环境实例（通常使用 env.unwrapped）
        center: 形变中心 [x, y, z] 世界坐标（米）
        radius: 影响区域半径（米）
        velocity_magnitude: 速度大小（米/秒）
        velocity_direction: 速度方向 [dx, dy, dz]（会被自动归一化）
        falloff_type: 衰减类型 - 'uniform', 'gaussian', 或 'linear'

    返回:
        包含形变元数据的字典:
            - center: 形变中心坐标
            - radius: 影响半径
            - velocity_magnitude: 速度大小
            - velocity_direction: 归一化后的速度方向
            - mean_velocity_change: 平均速度变化
            - max_velocity_change: 最大速度变化
            - affected_points: 受影响的点数
            - total_points: 总点数
            - falloff_type: 衰减类型
    """
    device = env.renderer.device

    # 1. 归一化方向向量
    velocity_direction = np.array(velocity_direction, dtype=np.float32)
    velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)

    # 2. 获取当前状态
    state = env.renderer.get_state()
    all_points = state['x']      # (N_total, 3) 所有点的位置（物体+机器人）
    velocities = state['v']      # (N_object, 3) 只有物体点的速度

    # 只对物体点计算权重（前N_object个点）
    num_object_points = velocities.shape[0]
    object_points = all_points[:num_object_points]  # (N_object, 3)

    # 3. 计算空间权重
    center_tensor = torch.tensor(center, dtype=torch.float32, device=device)
    weights = compute_spatial_weights(
        object_points, center_tensor, radius, falloff_type
    )

    # 4. 计算速度变化
    direction_tensor = torch.tensor(
        velocity_direction, dtype=torch.float32, device=device
    )
    velocity_delta = weights.unsqueeze(1) * direction_tensor * velocity_magnitude

    # 5. 应用速度变化
    new_velocities = velocities + velocity_delta
    state['v'] = new_velocities

    # 6. 更新渲染器状态
    env.renderer.update_state(state)

    # 7. 直接更新物理引擎的速度状态（不调用reset，避免重新初始化）
    # 获取物理模拟器
    simulator = env.physics.dynamics_module.simulator

    # 获取当前物理引擎的位置（需要完整的物理点，不只是渲染点）
    # 物理引擎维护自己的状态，我们需要获取它的当前位置
    current_physics_x = simulator.wp_state.wp_x.numpy()  # Warp数组转numpy
    current_physics_x_torch = torch.from_numpy(current_physics_x).to(device)

    # 直接设置物理引擎的位置和速度
    simulator.set_init_state(
        x=current_physics_x_torch,  # 位置保持不变
        v=new_velocities  # 使用新的速度
    )

    # 8. 计算统计信息
    affected_points = (weights > 0).sum().item()

    if affected_points > 0:
        mean_velocity_change = velocity_delta[weights > 0].norm(dim=1).mean().item()
        max_velocity_change = velocity_delta[weights > 0].norm(dim=1).max().item()
    else:
        mean_velocity_change = 0.0
        max_velocity_change = 0.0

    return {
        'center': center if isinstance(center, list) else center.tolist(),
        'radius': radius,
        'velocity_magnitude': velocity_magnitude,
        'velocity_direction': velocity_direction.tolist(),
        'mean_velocity_change': mean_velocity_change,
        'max_velocity_change': max_velocity_change,
        'affected_points': int(affected_points),
        'total_points': num_object_points,
        'falloff_type': falloff_type
    }
