# -*- coding: utf-8 -*-
"""Verbatim legacy function bodies. See PROJECT_MIT_ORIGINAL.txt.
Physical claims in historical comments are unverified; this is a construction diagnostic.
"""
import numpy as np
AR_PHI=0.92

def generate_ar1_timeseries(profile, event_dt, n_steps=24, center_step=12,
                            phi=AR_PHI, noise_scale=0.03, year=2021):
    """
    V4 关键: AR(1) 平滑时序 → 保证 lag-1 ≈ φ ≈ 0.92

    模型:
        x(t) = μ(t) + z(t)
        z(t) = φ * z(t-1) + ε(t),  ε ~ N(0, σ²)
        μ(t) = Profile[dow, slot(t)] × (1 + year_drift)

    这样生成的时序:
      - 基线跟随 Profile 日内波动 → 真实日内模式
      - AR(1) 残差 → lag-1 自相关 ≈ φ ≈ 0.92
      - 年际漂移 → 不同年份数据有适度差异
    """
    dow = event_dt.weekday()
    center_slot = event_dt.hour * 4 + event_dt.minute // 15

    # Profile 基线序列 μ(t)
    mu = np.zeros(n_steps)
    for i in range(n_steps):
        offset_slot = center_slot - center_step + i
        d = dow
        if offset_slot < 0:
            d = (dow - 1) % 7
            offset_slot = offset_slot % 96
        elif offset_slot >= 96:
            d = (dow + 1) % 7
            offset_slot = offset_slot % 96
        mu[i] = profile[d, offset_slot]

    # 年际漂移 (微小 trend)
    year_drift = (year - 2020) * np.random.uniform(-0.008, 0.008)
    mu = mu * (1 + year_drift)

    # AR(1) 彩色噪声
    sigma_eps = noise_scale * np.mean(mu)
    z = np.zeros(n_steps)
    z[0] = np.random.normal(0, sigma_eps)
    for t in range(1, n_steps):
        z[t] = phi * z[t - 1] + np.random.normal(0, sigma_eps * np.sqrt(1 - phi ** 2))

    volumes = mu + z
    volumes = np.clip(volumes, 3.0, None)

    return volumes, mu  # 返回扰动后序列和纯基线

def apply_ev_perturbation_v4(volumes, baseline, distance_deg, severity_level,
                              dispatch_sec=300, response_sec=600,
                              ev_start=8, ev_duration=4):
    """
    V4 改进:
      1. 基于真实 dispatch/response 时间校准扰动持续期
      2. 扰动幅度与 dispatch 速度相关 (快=高优→大扰动)
      3. AR(1) 一致的平滑恢复 (非突变)
      4. Laplace 脉冲在 EV 中心 → 更真实的峰值冲击
    """
    n_steps = len(volumes)
    perturbed = volumes.copy()
    ev_mask = np.zeros(n_steps)

    # 严重度 → 基线冲击系数
    severity_alpha = {
        0: 0.0,
        1: np.random.uniform(0.06, 0.15),
        2: np.random.uniform(0.14, 0.25),
        3: np.random.uniform(0.22, 0.36),
        4: np.random.uniform(0.32, 0.48),
        5: np.random.uniform(0.42, 0.58),
    }
    alpha = severity_alpha.get(int(severity_level), np.random.uniform(0.08, 0.18))

    # Dispatch速度校准: 快速dispatch → 更紧急 → 更大扰动
    if dispatch_sec > 0:
        urgency_factor = np.clip(1.0 + (300 - dispatch_sec) / 600, 0.7, 1.5)
    else:
        urgency_factor = 1.0
    alpha = alpha * urgency_factor

    # 空间衰减
    sigma_spatial = 0.015
    spatial_decay = np.exp(-distance_deg / sigma_spatial)
    impact = alpha * spatial_decay

    ev_end = min(ev_start + ev_duration, n_steps)

    # Phase 1: EV 影响期间 (Laplace脉冲 + 抛物线包络)
    for i in range(ev_start, ev_end):
        progress = (i - ev_start) / max(ev_duration, 1)
        # 时间包络: 快速上升 → 峰值 → 缓慢恢复
        temporal_envelope = 1 - (2 * progress - 1) ** 2
        # Laplace 脉冲 (中心位置有更尖锐的冲击)
        center_dist = abs(progress - 0.4) * ev_duration
        laplace_peak = np.exp(-center_dist * 1.5)
        combined_envelope = 0.6 * temporal_envelope + 0.4 * laplace_peak

        drop = impact * baseline[i] * combined_envelope
        perturbed[i] = perturbed[i] - drop
        ev_mask[i] = 1.0

    # Phase 2: 平滑恢复 (指数+AR一致)
    if ev_end < n_steps and ev_end > 0:
        tau = max(2.0, response_sec / 900)  # 基于响应时间的恢复常数
        for i in range(ev_end, n_steps):
            target = volumes[i]
            t_since = i - ev_end
            recovery = 1 - np.exp(-t_since / tau)
            perturbed[i] = perturbed[ev_end - 1] + (target - perturbed[ev_end - 1]) * recovery

    perturbed = np.clip(perturbed, 3.0, volumes.max() * 1.3)
    return perturbed, ev_mask
