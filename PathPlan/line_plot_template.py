import matplotlib.pyplot as plt
import numpy as np

# Clear any existing settings
plt.rcdefaults()

# Set up clean plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': False
})

# Data - Replace with your actual data
x_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Your x-axis data points
y_values = [10, 25, 45, 78, 120, 165, 210, 85]  # Your y-axis data points
x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Custom labels for x-axis (optional)

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')

# Create line plot with markers
line = ax.plot(x_values, y_values, marker='o', linewidth=3, markersize=8, 
               color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='black', 
               markeredgewidth=1.5, alpha=0.8)

# Set labels and title - Customize these
ax.set_xlabel('X-Axis Label', fontweight='bold')
ax.set_ylabel('Y-Axis Label', fontweight='bold')
ax.set_title('Your Plot Title Here', fontweight='bold', pad=20)

# Set x-axis ticks and labels
ax.set_xticks(x_values)
ax.set_xticklabels(x_labels)

# Add value labels on markers
for i, (x, y) in enumerate(zip(x_values, y_values)):
    ax.annotate(str(y), (x, y), 
                textcoords="offset points", xytext=(0,12), ha='center',
                fontweight='bold', fontsize=12, color='#2E86AB')

# Customize appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Set axis limits
ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
ax.set_ylim(0, max(y_values) * 1.15)

# Add a subtle grid for better readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add footnote - Customize or remove as needed
ax.text(0.6, 0.2, 'Optional footnote text here', transform=ax.transAxes,
        fontsize=14, style='italic', alpha=0.7)

# Adjust layout with more padding to prevent label cropping
plt.tight_layout(pad=2.0)

# Save the figure - Customize filename
plt.savefig('line_plot.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)

plt.show()

print("Line plot figure saved as 'line_plot.pdf'")

# Optional: Alternative color schemes
# Color scheme 1 (Blue-Purple): color='#2E86AB', markerfacecolor='#A23B72'
# Color scheme 2 (Green-Teal): color='#52B788', markerfacecolor='#2D6A4F'  
# Color scheme 3 (Orange-Red): color='#FB8500', markerfacecolor='#E85D04'
# Color scheme 4 (Navy-Gold): color='#003049', markerfacecolor='#FFB700'