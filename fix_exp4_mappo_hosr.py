"""
Quick fix: Exp4 MAPPO HOSR - 1 run per scenario, no repeats.
Updates exp4_data.json with real HOSR values.
"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
from uav_system.config import RESULT_DIR
import torch

MODEL_PATH = os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav.pt')

# Exp4 scenario configs: ALL use num_bs=8 (from experiments.py line 2073-2084),
# only num_uav varies per scenario
SCENARIOS = {
    'smart_city':           {'num_bs': 8,  'num_uav': 400},
    'industrial_inspection': {'num_bs': 8,  'num_uav': 300},
    'agriculture':          {'num_bs': 8,  'num_uav': 350},
    'emergency_rescue':     {'num_bs': 8,  'num_uav': 150},
    'logistics_delivery':    {'num_bs': 8,  'num_uav': 500},
}

SEED = 30042  # same as original exp4


def evaluate_mappo_scenario(scenario_name, cfg, seed=SEED, num_steps=150):
    """Run 1 MAPPO evaluation for a given scenario, return stats dict with real HOSR."""
    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}")
        return None

    env = MultiAgentHandoverEnv(
        num_bs=cfg['num_bs'], num_uav=cfg['num_uav'],
        max_steps=num_steps, seed=seed,
        bs_capacity_range=(500, 1000), pos_range=1000,
        event_probability=0.05,
    )
    obs_dict, global_state = env.reset()

    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
    )
    try:
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
    except Exception as e:
        print(f"  Load failed: {e}")
        return None

    total_attempts = 0
    total_success = 0
    total_rollback = 0
    total_disconnect = 0

    for step in range(num_steps):
        biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                     for uid in range(env.num_agents)}
        actions, _, _, _, _ = agent.select_actions(
            obs_dict, global_state, biz_types=biz_types, training=False)
        obs_dict, global_state, rewards, team_reward, done, info = env.step(actions)

        diag = info.get('reward_diag', {})
        total_attempts += diag.get('switch_attempts', 0)
        total_success += diag.get('switch_success', 0)
        total_rollback += diag.get('switch_rollback', 0)
        total_disconnect += diag.get('switch_disconnect', 0)

    final_sats = [env.env.uavs[uid].current_satisfaction for uid in range(env.num_agents)]
    connected_count = sum(1 for uid in range(env.num_agents) 
                          if env.env.uavs[uid].connected_bs_id is not None)

    real_hosr = total_success / total_attempts if total_attempts > 0 else 1.0

    stats = {
        'avg_satisfaction': np.mean(final_sats),
        'critical_satisfaction': np.mean([s for i, s in enumerate(final_sats)
                                          if env.env.uavs[i].true_business_type.value == 0]),
        'connected_ratio': connected_count / max(env.num_agents, 1),
        'handover_success_rate': real_hosr,
        'load_variance': np.var([bs.load_ratio for bs in env.env.base_stations.values()]),
        '_attempts': total_attempts,
        '_success': total_success,
        '_rollback': total_rollback,
        '_disconnect': total_disconnect,
    }

    print(f"  [{scenario_name}] HOSR={real_hosr*100:.1f}% "
          f"(ok={total_success}/try={total_attempts}, rollback={total_rollback}, disc={total_disconnect})")
    return stats


def main():
    print("=" * 60)
    print("  Exp4 MAPPO HOSR Quick Fix (1 run per scenario)")
    print("=" * 60)

    exp4_path = os.path.join('experiment_results', 'exp4_data.json')
    with open(exp4_path, 'r', encoding='utf-8') as f:
        exp4_data = json.load(f)

    results = {}
    for name, cfg in SCENARIOS.items():
        if name not in exp4_data or 'mappo' not in exp4_data[name]:
            print(f"\n  Skipping {name} (no mappo data)")
            continue
        print(f"\n[{name}] n_bs={cfg['num_bs']}, n_uav={cfg['num_uav']}")
        result = evaluate_mappo_scenario(name, cfg)
        if result:
            results[name] = result

    # Update JSON
    print("\n" + "=" * 60)
    print("  Updating exp4_data.json...")
    for name, stats in results.items():
        old = exp4_data[name]['mappo']['handover_success_rate'][0]
        exp4_data[name]['mappo']['handover_success_rate'] = [stats['handover_success_rate'], 0.0]
        print(f"  {name}: {old*100:.0f}% -> {stats['handover_success_rate']*100:.1f}%")

    # Also update avg_satisfaction and critical_satisfaction to be consistent
    for name, stats in results.items():
        exp4_data[name]['mappo']['avg_satisfaction'] = [float(stats['avg_satisfaction']), 0.0]
        exp4_data[name]['mappo']['critical_satisfaction'] = [float(stats['critical_satisfaction']), 0.0]
        exp4_data[name]['mappo']['connected_ratio'] = [float(stats['connected_ratio']), 0.0]
        exp4_data[name]['mappo']['load_variance'] = [float(stats['load_variance']), 0.0]

    with open(exp4_path, 'w', encoding='utf-8') as f:
        json.dump(exp4_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Done! Saved to {exp4_path}")

    # Summary table
    print(f"\n  {'Scenario':<22} {'HOSR':>8} {'Sat':>8} {'CritSat':>8} {'ConnRate':>9} {'LoadVar(x1e3)':>13}")
    print(f"  {'-'*70}")
    for name in SCENARIOS:
        if name in results:
            r = results[name]
            print(f"  {name:<22} {r['handover_success_rate']*100:>7.1f}% "
                  f"{r['avg_satisfaction']:>8.3f} {r['critical_satisfaction']:>8.3f} "
                  f"{r['connected_ratio']*100:>8.1f}% {r['load_variance']*1000:>12.3f}")


if __name__ == '__main__':
    main()
