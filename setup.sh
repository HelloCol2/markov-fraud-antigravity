#!/bin/bash
# Quick setup script for Antigravity Defender

echo "🧲 Antigravity Defender - Quick Setup"
echo "======================================"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q gym==0.26.2 stable-baselines3==2.2.1 numpy pandas matplotlib seaborn tqdm scipy

# Generate synthetic data
echo ""
echo "📊 Generating synthetic fraud data (2000 episodes)..."
python env/synth_data.py

# Quick test
echo ""
echo "🧪 Running environment test..."
python -c "from env.fraud_env import FraudAntigravityEnv; env = FraudAntigravityEnv(); obs = env.reset(); print('✅ Environment works!'); print(f'   Fraudster obs: {obs[0].shape}'); print(f'   Defender obs: {obs[1].shape}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Train: python training/train_marl.py --test-mode"
echo "  2. Evaluate: python training/evaluate.py"
echo "  3. Visualize: python utils/visualize.py"
