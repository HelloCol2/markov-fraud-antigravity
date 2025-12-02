"""
Interactive Process Visualizer
Generates visual diagrams of the Antigravity Defender system
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_system_architecture_diagram():
    """Create system architecture visualization"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🧲 Antigravity Defender System Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Data layer
    data_box = FancyBboxPatch((0.5, 8), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.5, 8.4, 'Dataset\n200k samples', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Environment layer
    env_box = FancyBboxPatch((3.5, 8), 3, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(env_box)
    ax.text(5, 8.4, 'FraudAntigravityEnv\nMarkov Game', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Agent layer
    fraudster_box = FancyBboxPatch((1, 6), 2.5, 1.2,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='red', facecolor='#ffcccc', linewidth=2)
    ax.add_patch(fraudster_box)
    ax.text(2.25, 6.8, 'Fraudster Agent', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.25, 6.4, 'PPO Network\n[10] → [64,64] → [3]', ha='center', fontsize=8)
    
    defender_box = FancyBboxPatch((6.5, 6), 2.5, 1.2,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='darkblue', facecolor='#cce5ff', linewidth=2)
    ax.add_patch(defender_box)
    ax.text(7.75, 6.8, 'Antigravity Defender', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.75, 6.4, 'Enhanced PPO\n[12] → [256,256,128] → [3]', ha='center', fontsize=8)
    
    # Training modules
    train_box = FancyBboxPatch((3, 4), 4, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='#e6ccff', linewidth=2)
    ax.add_patch(train_box)
    ax.text(5, 4.7, 'Training Pipeline', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 4.3, 'Phase 1: Pre-train | Phase 2: Co-train', ha='center', fontsize=9)
    
    # Evaluation
    eval_box = FancyBboxPatch((1.5, 2), 2, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='#ffe6cc', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(2.5, 2.6, 'Evaluation', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.5, 2.2, 'vs Baselines', ha='center', fontsize=9)
    
    # Visualization
    viz_box = FancyBboxPatch((6.5, 2), 2, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(7.5, 2.6, 'Visualization', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 2.2, 'Learning Curves', ha='center', fontsize=9)
    
    # Results
    result_box = FancyBboxPatch((3.5, 0.2), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='gold', facecolor='#ffffcc', linewidth=3)
    ax.add_patch(result_box)
    ax.text(5, 0.6, '🏆 Results: 17% Fraud Success, 5.5 System Loss', 
            ha='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', lw=2, color='black')
    
    # Data → Env
    ax.annotate('', xy=(3.5, 8.4), xytext=(2.5, 8.4), arrowprops=arrow_style)
    
    # Env → Agents
    ax.annotate('', xy=(2.25, 7.2), xytext=(4.5, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(7.75, 7.2), xytext=(5.5, 8), arrowprops=arrow_style)
    
    # Agents ↔ Training (bidirectional)
    ax.annotate('', xy=(4, 5), xytext=(2.25, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 5), xytext=(7.75, 6), arrowprops=arrow_style)
    
    # Training → Eval & Viz
    ax.annotate('', xy=(2.5, 3), xytext=(4.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 3), xytext=(5.5, 4), arrowprops=arrow_style)
    
    # Eval & Viz → Results
    ax.annotate('', xy=(4.5, 1), xytext=(2.5, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(5.5, 1), xytext=(7.5, 2), arrowprops=arrow_style)
    
    plt.tight_layout()
    plt.savefig('analysis_output/system_architecture.png', dpi=300, bbox_inches='tight')
    print('✅ Saved: analysis_output/system_architecture.png')
    plt.close()


def create_training_flow_diagram():
    """Create training process flow"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(5, 11.5, '🔄 Training Flow: Antigravity Defender', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Phase 1
    ax.text(5, 10.5, 'PHASE 1: Pre-Training (Episodes 1-1000)', 
            ha='center', fontsize=13, fontweight='bold', color='blue')
    
    steps_phase1 = [
        ('Initialize\nDefender & Oracle', 9.5),
        ('Collect 2048 steps', 8.5),
        ('Compute GAE advantages', 7.5),
        ('Update policy (15 epochs)', 6.5),
        ('Evaluate performance', 5.5),
    ]
    
    for i, (text, y) in enumerate(steps_phase1):
        box = FancyBboxPatch((2, y-0.3), 6, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='blue', facecolor='lightblue', linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, y, text, ha='center', va='center', fontsize=10)
        
        if i < len(steps_phase1) - 1:
            ax.annotate('', xy=(5, y-0.5), xytext=(5, y-0.3),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Phase 2
    ax.text(5, 4.5, 'PHASE 2: Adversarial Co-Training (Episodes 1001-2000)', 
            ha='center', fontsize=13, fontweight='bold', color='red')
    
    # Round visualization
    for round_idx in range(3):
        y_start = 3.5 - round_idx * 1.2
        
        # Fraudster training
        box1 = FancyBboxPatch((1, y_start-0.2), 3.5, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='#ffcccc', linewidth=1.5)
        ax.add_patch(box1)
        ax.text(2.75, y_start, f'Round {round_idx+1}: Train Fraudster', 
                ha='center', va='center', fontsize=9)
        
        # Defender training
        box2 = FancyBboxPatch((5.5, y_start-0.2), 3.5, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor='darkblue', facecolor='#cce5ff', linewidth=1.5)
        ax.add_patch(box2)
        ax.text(7.25, y_start, f'Train Defender', 
                ha='center', va='center', fontsize=9)
        
        # Arrow between
        ax.annotate('', xy=(5.5, y_start), xytext=(4.5, y_start),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
    
    # Nash equilibrium
    nash_box = FancyBboxPatch((3, 0.5), 4, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor='gold', facecolor='#ffffcc', linewidth=2)
    ax.add_patch(nash_box)
    ax.text(5, 0.8, '⚖️ Nash Equilibrium Reached', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis_output/training_flow.png', dpi=300, bbox_inches='tight')
    print('✅ Saved: analysis_output/training_flow.png')
    plt.close()


def create_decision_flowchart():
    """Create antigravity decision process flowchart"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(5, 13.5, '🎯 Antigravity Decision Process', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((3, 12), 4, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 12.4, 'INPUT:\nObservation Vector [12 features]', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Decision 1: Payoff trend
    decision1 = FancyBboxPatch((2.5, 10), 5, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='#ffe6e6', linewidth=2)
    ax.add_patch(decision1)
    ax.text(5, 10.7, 'Fraudster Payoff\nTrending Up?', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 10.2, '(payoff_trend > 0.3 AND\nfraud_rate > 0.4)', 
            ha='center', va='center', fontsize=8)
    
    # YES path
    action_strict = FancyBboxPatch((7, 8.5), 2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='darkred', facecolor='#ff9999', linewidth=2)
    ax.add_patch(action_strict)
    ax.text(8, 8.9, 'STRICT (2)\nCounter-Force!', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # NO path - Decision 2
    decision2 = FancyBboxPatch((1, 7.5), 3, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='orange', facecolor='#fff2e6', linewidth=2)
    ax.add_patch(decision2)
    ax.text(2.5, 8.2, 'High FP Rate OR\nLow Budget?', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # YES - Lenient
    action_lenient = FancyBboxPatch((0.5, 6), 2, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='green', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(action_lenient)
    ax.text(1.5, 6.3, 'LENIENT (0)\nEfficiency', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # NO - Threat calculation
    threat_calc = FancyBboxPatch((3.5, 5.5), 3, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='#f2e6ff', linewidth=2)
    ax.add_patch(threat_calc)
    ax.text(5, 6.4, 'Calculate\nThreat Score', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.9, '(risk + amount +\n2*fraud_rate) / 4', 
            ha='center', va='center', fontsize=8)
    
    # Threat outcomes
    outcomes = [
        ('threat > 0.65', 'STRICT (2)', 3.5, '#ffe6e6'),
        ('0.35 < threat < 0.65', 'NORMAL (1)', 2.3, '#e6f2ff'),
        ('threat ≤ 0.35', 'LENIENT (0)', 1.1, '#e6ffe6'),
    ]
    
    for i, (condition, action, y, color) in enumerate(outcomes):
        x_offset = 7 if i == 0 else (5 if i == 1 else 2.5)
        box = FancyBboxPatch((x_offset, y), 2, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_offset+1, y+0.4, condition, ha='center', fontsize=8)
        ax.text(x_offset+1, y+0.15, action, ha='center', fontsize=9, fontweight='bold')
    
    # Final output
    output_box = FancyBboxPatch((3.5, 0), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='gold', facecolor='#ffffcc', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.3, 'OUTPUT: Defense Action', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrow = dict(arrowstyle='->', lw=1.5, color='black')
    ax.annotate('', xy=(5, 11), xytext=(5, 12), arrowprops=arrow)
    ax.annotate('YES', xy=(7.5, 9.3), xytext=(6.5, 10), arrowprops=arrow, fontsize=9, color='red')
    ax.annotate('NO', xy=(2.5, 8.5), xytext=(4, 10), arrowprops=arrow, fontsize=9)
    ax.annotate('YES', xy=(1.5, 6.6), xytext=(2, 7.5), arrowprops=arrow, fontsize=8)
    ax.annotate('NO', xy=(5, 6.7), xytext=(3, 7.5), arrowprops=arrow, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('analysis_output/decision_flowchart.png', dpi=300, bbox_inches='tight')
    print('✅ Saved: analysis_output/decision_flowchart.png')
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('analysis_output', exist_ok=True)
    
    print('🎨 Generating process visualizations...\n')
    
    create_system_architecture_diagram()
    create_training_flow_diagram()
    create_decision_flowchart()
    
    print('\n✅ All visualizations created!')
    print('   Check analysis_output/ for PNG files')
