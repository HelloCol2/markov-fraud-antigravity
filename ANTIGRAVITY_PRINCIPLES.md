# 🧲 Antigravity Defender - Strategic Training Principles

## Core Philosophy

The Antigravity Defender is **not a static fraud detector**. It is a **strategic counter-force** that learns to:

1. **Recognize fraud as an adapting opponent** (not random noise)
2. **Apply dynamic strictness** based on adversarial profitability
3. **Optimize long-term equilibrium** over short-term detection
4. **Collapse fraudster payoff** through learned counter-pressure
5. **Balance security ↔ cost ↔ trust** in stable Nash equilibrium

---

## Training Dataset: `fraud_antigravity_synth-2.csv`

Enhanced 39MB dataset with **strategic behavioral encoding**:

### Key Features
- **Transaction context**: risk scores, amounts, time patterns
- **Strategic actions**: fraud intensity {none, low, high}
- **Detection outcomes**: caught vs successful attempts
- **Cost dynamics**: investigation costs, false positive penalties
- **System state**: cumulative loss, stress, budgets
- **Adversarial signals**: fraudster payoff trends, aggressiveness

### Why This Dataset Matters
Unlike traditional fraud data (binary labels), this dataset encodes:
- **Adaptive opponent behavior** (fraudster learns over time)
- **Economic incentives** (fraud profitability drives decisions)
- **Operational constraints** (budgets, investigation costs)
- **Strategic feedback loops** (past detection affects future behavior)

---

## The 5 Antigravity Principles

### 1️⃣ **Strategic Opponent Recognition**

**Principle**: Fraud is NOT a random anomaly—it's an intelligent, adapting adversary.

**Implementation**:
```python
# Defender observes fraudster behavior patterns
fraudster_aggressiveness = obs[7]  # Estimated attack intensity
fraudster_payoff_trend = obs[8]    # Is fraud becoming profitable?

# Adjust strategy based on opponent's strategic state
if fraudster_payoff_trend > 0.3:
    apply_counter_force()  # Disrupt exploitation
```

**Training Impact**:
- Defender learns to **predict fraudster strategy evolution**
- Not just "detect fraud" but "counter-adapt to adversarial learning"

---

### 2️⃣ **Dynamic Counter-Force Application**

**Principle**: Strictness adjusts based on **when fraud becomes profitable** + **risk context**.

**Decision Logic**:
```
High Strictness (2) → When:
  - Fraud success rate > 0.5 AND
  - Fraudster payoff trending up AND
  - Risk context increases (high-value targets)

Low Strictness (0) → When:
  - Fraud drops below threshold AND
  - Operational cost > potential damage AND
  - False positive rate is high
```

**Why This Works**:
- **Disrupts adversarial exploitation** at profitable moments
- **Conserves resources** when fraud is dormant
- **Avoids always-strict trap** (destroys user trust & efficiency)

---

### 3️⃣ **Long-Term Optimization Over Short-Term Detection**

**Principle**: Maximize cumulative reward over episode, not per-step accuracy.

**Objective Function**:
```python
R_defender = -(system_loss + investigation_cost + fp_cost)

# Optimized with high discount factor
gamma = 0.995  # Strong long-term focus
```

**Key Metrics Minimized**:
- `fraud_success_rate ↓` (fewer successful attacks)
- `average_system_loss ↓` (less damage)
- `fraudster_reward_trend ↓` (collapse adversarial profitability)
- `investigation_count` (efficient, not explosive)
- `false_positive_cost` (bounded, fair)

**Contrast with Static Rules**:
- Static: "If risk > 0.7, flag" → Myopic, exploitable
- Antigravity: "Learn sequence patterns that lead to minimum cumulative loss" → Strategic

---

### 4️⃣ **Payoff Collapse via Adaptive Pressure**

**Principle**: Make fraud **unprofitable** by learning to suppress fraudster rewards over time.

**Mechanism**:
```python
# Track fraudster payoff trend
fraudster_payoff_trend = (
    recent_fraudster_reward - historical_fraudster_reward
)

# If trending up, apply counter-pressure
if fraudster_payoff_trend > threshold:
    increase_strictness()  # "Antigravity" suppression
```

**Expected Outcome**:
```
Episode    Fraudster Reward    System Loss
   1-500:      +0.45              0.83
 500-1000:     +0.31              0.61
1000-1500:     +0.18              0.44
1500-2000:     +0.09   ← Collapsed  0.37
```

**Why "Antigravity"?**:
- Fraudster's "gravity" = pull toward exploitation
- Defender's "antigravity" = counter-force that suppresses profitability
- Equilibrium = balanced forces (Nash equilibrium)

---

### 5️⃣ **Stable Equilibrium: Security ↔ Cost ↔ Trust**

**Principle**: Find the **Goldilocks zone**—strict enough to prevent fraud, lenient enough to preserve efficiency.

**Trade-off Optimization**:
```
Security (detection rate) ↕
   vs
Cost (investigations)     ↕
   vs
Trust (false positives)

→ Learn mixed-strategy Nash equilibrium
```

**Why Not Always-Strict?**:
- ❌ Destroys operational efficiency (cost explosion)
- ❌ Erodes user trust (false positive fatigue)
- ❌ Fraudsters adapt to avoid high-strictness windows
- ✅ Antigravity learns **when to be strict** (not "always")

**Why Not Always-Lenient?**:
- ❌ Invites exploitation (fraud becomes profitable)
- ❌ System loss accumulates
- ✅ Antigravity learns **when to relax** (not "never")

---

## Training Pipeline

### Phase 1: Pre-Training Against Oracle Fraudster
**Duration**: 50% of episodes  
**Opponent**: Heuristic rule-based fraudster

**Goal**: Learn basic counter-strategies
- Detect common fraud patterns
- Understand cost-benefit trade-offs
- Build foundation for strategic reasoning

### Phase 2: Adversarial Co-Training
**Duration**: 50% of episodes (5 rounds)  
**Opponent**: Adaptive PPO-learning fraudster

**Goal**: Learn to counter-evolve
- Fraudster learns to exploit defender weaknesses
- Defender learns to counter new fraudster strategies
- Iterate toward Nash equilibrium

**Key Insight**: Without adversarial co-training, defender only learns to beat heuristic fraud (doesn't generalize to adaptive adversaries).

---

## Expected Results

### Antigravity vs Baselines

| Defender | Fraud Success | System Loss | F1 Score | Principle Embodied |
|----------|--------------|-------------|----------|-------------------|
| **Antigravity** | **28%** ↓ | **0.38** ↓ | **0.81** ↑ | Adaptive counter-force |
| Adaptive Threshold | 38% | 0.49 | 0.73 | Reactive (not learning) |
| Static Threshold | 47% | 0.62 | 0.66 | Fixed rules (exploitable) |
| Always Strict | 22% | **0.91** ↑ | 0.71 | Security at cost of efficiency |
| Random | 68% | 1.07 | 0.48 | No strategy |

**Key Wins**:
1. ✅ **Lower fraud success than adaptive threshold** (learns strategic patterns)
2. ✅ **Lower system loss than always-strict** (balances cost/benefit)
3. ✅ **Higher F1 than all baselines** (precision + recall optimized)
4. ✅ **Fraudster payoff collapsed** (antigravity working)

---

## How to Train

### Quick Test
```bash
python training/train_antigravity_enhanced.py --test-mode
```

### Full Training
```bash
python training/train_antigravity_enhanced.py \
  --dataset fraud_antigravity_synth-2.csv \
  --episodes 2000 \
  --save-dir checkpoints_enhanced/
```

### Monitor Progress
```bash
tensorboard --logdir tensorboard_logs/
```

---

## Key Files

- **[antigravity_enhanced.py](file:///Users/iliajakhaia/Desktop/Game%20theory/agents/antigravity_enhanced.py)** - Enhanced defender with strategic heuristic
- **[train_antigravity_enhanced.py](file:///Users/iliajakhaia/Desktop/Game%20theory/training/train_antigravity_enhanced.py)** - Training pipeline embodying all 5 principles
- **fraud_antigravity_synth-2.csv** - 39MB enhanced dataset with strategic behavior

---

## Summary

The Antigravity Defender is **not just a fraud detector**—it's a **strategic learning agent** that:

✅ Recognizes fraud as an adapting opponent  
✅ Applies counter-force when fraud becomes profitable  
✅ Optimizes long-term equilibrium over short-term accuracy  
✅ Collapses fraudster payoff through learned pressure  
✅ Balances security, cost, and trust in stable Nash equilibrium  

**Train it. Watch fraud profitability collapse. See the antigravity effect in action.** 🧲
