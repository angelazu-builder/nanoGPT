#!/opt/miniconda3/bin/python3
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define output image paths
day3_output = "Day 3/model_championship_text_samples.png"
day2_output = "Day 2/model_championship_text_samples.png"
root_output = "runs/model_championship_text_samples.png"

# Setup figure canvas (High Resolution Document Card)
fig = plt.figure(figsize=(14, 12), dpi=300)
fig.patch.set_facecolor('#0f172a') # Dark slate background

# Main Title
plt.text(0.5, 0.95, "NanoGPT Best Checkpoint Generation Championship", 
         fontsize=18, fontweight='bold', color='#f8fafc', ha='center', va='center')
plt.text(0.5, 0.92, "Side-by-Side Text Quality Audit across 3 Major Model Milestones", 
         fontsize=12, color='#94a3b8', ha='center', va='center')

# Define Card Data
cards = [
    {
        "title": "[1] exp05: Character-Level Baseline (block_size=64)",
        "meta": "Best Step 4500 | Val Loss: 1.4922 | 2.15 BPC | Gibberish: ~1.5%",
        "color": "#1e3a8a",
        "border": "#3b82f6",
        "text_color": "#dbeafe",
        "sample": """ISABELLA:
So it is the state as you have been,
By report the of the cordial severest hands
With the fine incensed better than dead.

QUEEN MARGARET:
I cannot forth your grace: good no more to you,
But leave me to the world."""
    },
    {
        "title": "[2] exp06: Character-Level Context Expansion (block_size=256)",
        "meta": "Best Step 2100 | Val Loss: 1.4668 | 2.12 BPC | Gibberish: ~1.2%",
        "color": "#064e3b",
        "border": "#10b981",
        "text_color": "#d1fae5",
        "sample": """All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death."""
    },
    {
        "title": "[3] exp07: Subword BPE Champion (block_size=128, V=50,257)",
        "meta": "Best Step 900 | Val Loss: 4.8475 | 2.12 BPC (Winner) | 0.0% Gibberish",
        "color": "#7c2d12",
        "border": "#f97316",
        "text_color": "#ffedd5",
        "sample": """ISABELLA:
I pray you, go, sir; you are the first.

ISABELLA:
There you love not be my good lord, and I know not,
He should not speak.

DUKE VINCENTIO:
What's enough.

Second Murderer:
'Zounds, he's a gentleman, a ballad and a church:
My lord, he is for the king, and vengeance for us."""
    }
]

# Draw Cards
y_positions = [0.65, 0.38, 0.09]

ax = fig.add_subplot(111)
ax.axis('off')

for idx, card in enumerate(cards):
    y_top = y_positions[idx]
    
    # Background Box
    rect = patches.FancyBboxPatch((0.04, y_top), 0.92, 0.22,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=card["color"], edgecolor=card["border"],
                                linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    
    # Title & Meta
    ax.text(0.07, y_top + 0.18, card["title"], fontsize=13, fontweight='bold', color='#ffffff')
    ax.text(0.07, y_top + 0.145, card["meta"], fontsize=10, fontweight='bold', color='#cbd5e1')
    
    # Code / Sample Box
    sample_rect = patches.FancyBboxPatch((0.06, y_top + 0.02), 0.88, 0.11,
                                         boxstyle="round,pad=0.01,rounding_size=0.01",
                                         facecolor='#020617', edgecolor='#334155',
                                         linewidth=1)
    ax.add_patch(sample_rect)
    
    # Sample Text
    ax.text(0.08, y_top + 0.115, card["sample"], fontsize=9.5, family='monospace', 
            color=card["text_color"], va='top', ha='left')

# Save Deliverable Images
plt.savefig(day3_output, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(day2_output, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(root_output, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"✨ Successfully generated Deliverable Screenshots:")
print(f"  - {day3_output}")
print(f"  - {day2_output}")
print(f"  - {root_output}")
