"""
Draw the reference graph from Pr. MESTARI's course support.
Nodes: S, A, B, C, D, G
Edges: S->A(1), S->B(4), A->C(2), A->D(5), B->D(1), C->G(5), D->G(3)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

def draw_reference_graph(output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    
    pos = {
        'S': (0.0, 1.5),
        'A': (1.8, 2.8),
        'B': (1.8, 0.2),
        'C': (3.3, 2.8),
        'D': (3.3, 0.8),
        'G': (5.0, 1.5),
    }

    
    h_vals = {'S': 7, 'A': 6, 'B': 5, 'C': 4, 'D': 2, 'G': 0}

    
    edges = [
        ('S', 'A', 1,  (-0.1,  0.18)),
        ('S', 'B', 4,  (-0.1, -0.18)),
        ('A', 'C', 2,  ( 0.0,  0.18)),
        ('A', 'D', 5,  ( 0.15,-0.12)),
        ('B', 'D', 1,  ( 0.0, -0.18)),
        ('C', 'G', 5,  ( 0.1,  0.18)),
        ('D', 'G', 3,  ( 0.1, -0.18)),
    ]

    
    for (u, v, cost, loff) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color='#2c3e50',
                lw=2.0,
                mutation_scale=18,
                shrinkA=22, shrinkB=22,
            ))
        
        mx = (x1 + x2) / 2 + loff[0]
        my = (y1 + y2) / 2 + loff[1]
        ax.text(mx, my, str(cost),
                fontsize=13, fontweight='bold',
                color='#e74c3c',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    
    node_colors = {
        'S': '#2980b9',  
        'G': '#27ae60',   
        'A': '#f0f0f0',
        'B': '#f0f0f0',
        'C': '#f0f0f0',
        'D': '#f0f0f0',
    }
    node_border = {
        'S': '#1a5f8a',
        'G': '#1a7a45',
        'A': '#7f8c8d',
        'B': '#7f8c8d',
        'C': '#7f8c8d',
        'D': '#7f8c8d',
    }
    text_colors = {'S': 'white', 'G': 'white', 'A': '#2c3e50', 'B': '#2c3e50', 'C': '#2c3e50', 'D': '#2c3e50'}

    r = 0.32
    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), r, color=node_colors[node],
                             ec=node_border[node], lw=2.5, zorder=5)
        ax.add_patch(circle)
        
        ax.text(x, y + 0.04, node, fontsize=15, fontweight='bold',
                color=text_colors[node], ha='center', va='center', zorder=6)
        
        ax.text(x, y - 0.54, f'h={h_vals[node]}',
                fontsize=10, color='#555555',
                ha='center', va='center', zorder=6,
                style='italic')

    
    legend_elements = [
        mpatches.Patch(facecolor='#2980b9', edgecolor='#1a5f8a', label='État initial (S)'),
        mpatches.Patch(facecolor='#27ae60', edgecolor='#1a7a45', label='État but (G)'),
        mpatches.Patch(facecolor='#f0f0f0', edgecolor='#7f8c8d', label='État intermédiaire'),
        mpatches.Patch(facecolor='white',   edgecolor='white',   label='Coût en rouge sur chaque arc'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              ncol=2, fontsize=9.5, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.06))

    ax.set_title("Graphe orienté pondéré — Support du module (Pr. MESTARI)\n"
                 "h(S)=7,  h(A)=6,  h(B)=5,  h(C)=4,  h(D)=2,  h(G)=0",
                 fontsize=12, color='#2c3e50', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Graph saved: {output_path}")

if __name__ == "__main__":
    draw_reference_graph("experiments/reference_graph.png")