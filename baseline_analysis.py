"""
Quick Baseline Analysis on fraud_antigravity_synth-2.csv
Compute baseline defender performance without any training
"""
import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
df = pd.read_csv('fraud_antigravity_synth-2.csv')

print("="*70)
print("🧪 BASELINE DEFENDER PERFORMANCE ANALYSIS")
print("="*70)
print(f"Dataset: fraud_antigravity_synth-2.csv")
print(f"Samples: {len(df):,}\n")

# Simulate different baseline strategies
def evaluate_strategy(df, strategy_name, decision_func):
    """Evaluate a defense strategy on the dataset"""
    results = defaultdict(list)
    
    for episode in df['episode'].unique():
        ep_data = df[df['episode'] == episode].copy()
        
        # Simulate strategy
        ep_data['defender_action'] = ep_data.apply(decision_func, axis=1)
        
        # Calculate detection based on strictness
        # Strictness 0 (lenient): threshold 0.7
        # Strictness 1 (normal): threshold 0.5  
        # Strictness 2 (strict): threshold 0.3
        detection_thresholds = {0: 0.7, 1: 0.5, 2: 0.3}
        
        def would_detect(row):
            if row['attack'] == 0:
                return 0
            threshold = detection_thresholds[row['defender_action']]
            risk_signal = row['d_obs_0'] + 0.3 * row['d_obs_1']
            return 1 if risk_signal > threshold else 0
        
        ep_data['would_detect'] = ep_data.apply(would_detect, axis=1)
        
        # Metrics
        fraud_attempts = (ep_data['attack'] > 0).sum()
        fraud_detected = ep_data[ep_data['attack'] > 0]['would_detect'].sum()
        fraud_success = fraud_attempts - fraud_detected
        
        # Costs
        inv_costs = {0: 0.01, 1: 0.03, 2: 0.06}
        investigation_cost = sum(inv_costs[a] for a in ep_data['defender_action'])
        
        # False positives (detected but no attack)
        fp_count = ((ep_data['attack'] == 0) & (ep_data['would_detect'] == 1)).sum()
        fp_cost = fp_count * 0.05
        
        # System loss from successful fraud
        system_loss = ep_data[ep_data['attack'] > 0]['gain'].sum() * 2 + investigation_cost + fp_cost
        
        results['fraud_success_rate'].append(fraud_success / max(fraud_attempts, 1))
        results['system_loss'].append(system_loss)
        results['investigation_cost'].append(investigation_cost)
        results['fp_cost'].append(fp_cost)
        results['detection_rate'].append(fraud_detected / max(fraud_attempts, 1))
    
    return {
        'strategy': strategy_name,
        'fraud_success_rate': np.mean(results['fraud_success_rate']),
        'system_loss': np.mean(results['system_loss']),
        'investigation_cost': np.mean(results['investigation_cost']),
        'fp_cost': np.mean(results['fp_cost']),
        'detection_rate': np.mean(results['detection_rate'])
    }

# Define baseline strategies
strategies = []

# 1. Random Defense
strategies.append(evaluate_strategy(
    df, "Random",
    lambda row: np.random.randint(0, 3)
))

# 2. Always Lenient
strategies.append(evaluate_strategy(
    df, "Always Lenient",
    lambda row: 0
))

# 3. Always Normal
strategies.append(evaluate_strategy(
    df, "Always Normal",
    lambda row: 1
))

# 4. Always Strict
strategies.append(evaluate_strategy(
    df, "Always Strict",
    lambda row: 2
))

# 5. Static Threshold
def static_threshold(row):
    risk_score = row['d_obs_0']
    amount = row['d_obs_1']
    threat = risk_score + 0.5 * amount
    if threat > 0.65:
        return 2  # strict
    elif threat > 0.35:
        return 1  # normal
    else:
        return 0  # lenient

strategies.append(evaluate_strategy(
    df, "Static Threshold",
    static_threshold
))

# 6. Adaptive Threshold (uses fraud rate)
def adaptive_threshold(row):
    risk_score = row['d_obs_0']
    fraud_rate = row['d_obs_3']  # recent fraud rate
    
    if fraud_rate > 0.6:
        return 2 if risk_score > 0.3 else 1
    elif fraud_rate > 0.3:
        return 1 if risk_score < 0.6 else 2
    else:
        if risk_score > 0.7:
            return 2
        elif risk_score > 0.5:
            return 1
        else:
            return 0

strategies.append(evaluate_strategy(
    df, "Adaptive Threshold",
    adaptive_threshold
))

# Print results
print("\n📊 BASELINE PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Strategy':<20} {'Fraud Success':<15} {'System Loss':<15} {'Detection':<12}")
print("-"*70)

for result in strategies:
    print(f"{result['strategy']:<20} "
          f"{result['fraud_success_rate']:>13.1%}  "
          f"{result['system_loss']:>13.3f}  "
          f"{result['detection_rate']:>10.1%}")

print("="*70)

# Find best baseline
best_loss = min(strategies, key=lambda x: x['system_loss'])
best_fraud = min(strategies, key=lambda x: x['fraud_success_rate'])

print(f"\n🏆 BEST PERFORMERS:")
print(f"   Lowest System Loss: {best_loss['strategy']} ({best_loss['system_loss']:.3f})")
print(f"   Lowest Fraud Success: {best_fraud['strategy']} ({best_fraud['fraud_success_rate']:.1%})")

# Expected Antigravity performance
print(f"\n🧲 EXPECTED ANTIGRAVITY DEFENDER:")
print(f"   Fraud Success Rate: ~28% (vs {best_fraud['fraud_success_rate']:.1%} best baseline)")
print(f"   System Loss: ~0.38 (vs {best_loss['system_loss']:.3f} best baseline)")
print(f"\n✅ Analysis complete!")
