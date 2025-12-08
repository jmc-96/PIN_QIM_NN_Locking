#%% Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from I_Plot_style import apply_custom_style  # Optional style
import os

#%% --- CHART CONFIGURATIONS FOR LOCKING ---
chart_configs = {
    "WA": {
        "data": [
            [96.48, 74.29, 50.65, 8.95, 9.89, 8.14],   # NN1
            [96.58, 92.67, 79.57, 16.21, 10.13, 8.49], # NN2
            [93.29, 92.28, 92.55, 85.16, 24.89, 9.54], # NN3
            [95.04, 94.92, 94.61, 84.99, 16.16, 9.53], # NN4
            [96.21, 95.8, 94.96, 38.64, 11.45, 10.79], # NN5
        ],
        "ylabel": "Accuracy (%)",
        "ylim": (0, 100),
        "save_filename": "chart_WA.png",
        "x_labels": ['0', '4', '10', '100', '1000', '10000'],
    },
    "WR": {
        "data": [
            [96.48, 96.39, 95.67, 84.74, 9.42],   # NN1
            [96.58, 96.57, 95.99, 87.46, 11.05],   # NN2
            [93.29, 93.34, 93.32, 91.18, 63.04],   # NN3
            [95.04, 95.09, 95.0, 66.13, 9.71],    # NN4
            [96.21, 96.09, 95.84, 77.31, 10.09],   # NN5
        ],
        "ylabel": "Accuracy (%)",
        "ylim": (0, 100),
        "save_filename": "chart_WR.png",
        "x_labels": ['0', '100', '1000', '10000', '81700'],
    },
    "BR": {
        "data": [
            [96.48, 96.26, 95.98, 95.47],  # NN1
            [96.58, 96.54, 96.23, 95.64],  # NN2
            [93.29, 93.26, 93.34, 93.39, 93.01],  # NN3
            [95.04, 95.02, 95.06, 93.44],  # NN4
            [96.21, 96.24, 95.84, 94.86],  # NN5
        ],
        "ylabel": "Accuracy (%)",
        "ylim": (0, 100),
        "save_filename": "chart_BR.png",
    },
}

def plot_WA_chart(cfg):
    data_to_plot = cfg["data"]
    ylabel = cfg["ylabel"]
    ylim = cfg["ylim"]
    save_filename = cfg["save_filename"]
    x_labels = cfg["x_labels"]

    x_numeric = [float(label) for label in x_labels]
    x_plot = [0 if val == 0 else np.log10(val) for val in x_numeric]

    y = np.array(data_to_plot).T
    n_networks = y.shape[1]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    formal_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

    for i in range(n_networks):
        x_interp = x_plot
        y_interp = y[:, i]
        x_smooth = np.linspace(min(x_interp), max(x_interp), 200)
        pchip = PchipInterpolator(x_interp, y_interp)
        y_smooth = pchip(x_smooth)
        ax.plot(x_smooth, y_smooth, color=formal_colors[i], linewidth=0.8, zorder=2)
        ax.plot(x_plot, y[:, i], color=formal_colors[i], marker='o', linestyle='None', markersize=2.5, label=f'NN{i+1}', zorder=3)

    for xi in x_plot:
        ax.axvline(x=xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r'$\log_{10} r$ (pseudo-$\log_{10} r$ | $r=0$)', labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    apply_custom_style(ax)
    ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, save_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # Save as EPS
    eps_save_path = os.path.splitext(save_path)[0] + ".eps"
    plt.savefig(eps_save_path, format='eps', bbox_inches='tight')
    plt.show()

def plot_BR_chart(cfg):
    # Define key lengths for each NN (order matches your data)
    key_lengths = [
        [0, 10, 100, 140],           # NN1
        [0, 10, 100, 160],           # NN2
        [0, 10, 100, 1000, 1560],    # NN3
        [0, 10, 100, 530],           # NN4
        [0, 10, 100, 240],           # NN5
    ]
    data_to_plot = cfg["data"]
    ylabel = cfg["ylabel"]
    ylim = cfg["ylim"]
    save_filename = cfg["save_filename"]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    formal_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

    all_x_plot = []
    all_x_labels = []
    for i, (x_vals, y_vals) in enumerate(zip(key_lengths, data_to_plot)):
        x_plot = [0 if val == 0 else np.log10(val) for val in x_vals]
        x_interp = x_plot
        y_interp = y_vals
        x_smooth = np.linspace(min(x_interp), max(x_interp), 200)
        pchip = PchipInterpolator(x_interp, y_interp)
        y_smooth = pchip(x_smooth)
        ax.plot(x_smooth, y_smooth, color=formal_colors[i], linewidth=0.8, zorder=2)
        ax.plot(x_plot, y_interp, color=formal_colors[i], marker='o', linestyle='None', markersize=2.5, label=f'NN{i+1}', zorder=3)
        all_x_plot.extend(x_plot)
        all_x_labels.extend([str(val) for val in x_vals])

    # Only show these ticks/labels on the x axis
    show_labels = ['0', '10', '100', '1560']
    show_x_plot = []
    for label in show_labels:
        for x, lab in zip(all_x_plot, all_x_labels):
            if lab == label:
                show_x_plot.append(x)
                break

    for xi in show_x_plot:
        ax.axvline(x=xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xticks(show_x_plot)
    ax.set_xticklabels(show_labels)
    ax.set_xlabel(r'$\log_{10} r$ (pseudo-$\log_{10} r$ | $r=0$)', labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_ylim(90, 100)  # Show only 90-100 on y axis
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=90)  # Optional, ensures lower border is at 90
    apply_custom_style(ax)
    ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, save_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # Save as EPS
    eps_save_path = os.path.splitext(save_path)[0] + ".eps"
    plt.savefig(eps_save_path, format='eps', bbox_inches='tight')
    plt.show()

def plot_WR_chart(cfg):
    data_to_plot = cfg["data"]
    ylabel = cfg["ylabel"]
    ylim = cfg["ylim"]
    save_filename = cfg["save_filename"]

    # Define per-NN x_labels for WR chart
    x_labels_list = [
        cfg["x_labels"],  # For NN1
        ['0', '100', '1000', '10000', '83900'],     # For NN2
        ['0', '100', '1000', '10000', '100000'],    # For NN3
        ['0', '100', '1000', '10000', '100000'],    # For NN4
        ['0', '100', '1000', '10000', '87700'],     # For NN5
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    formal_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

    n_networks = len(data_to_plot)
    all_x_plot = []
    all_x_labels = []
    for i in range(n_networks):
        x_labels = x_labels_list[i]
        x_numeric = [float(label) for label in x_labels]
        x_plot = [0 if val == 0 else np.log10(val) for val in x_numeric]
        y = data_to_plot[i]
        x_interp = x_plot
        y_interp = y
        x_smooth = np.linspace(min(x_interp), max(x_interp), 200)
        pchip = PchipInterpolator(x_interp, y_interp)
        y_smooth = pchip(x_smooth)
        ax.plot(x_smooth, y_smooth, color=formal_colors[i], linewidth=0.8, zorder=2)
        ax.plot(x_plot, y_interp, color=formal_colors[i], marker='o', linestyle='None', markersize=2.5, label=f'NN{i+1}', zorder=3)
        all_x_plot.extend(x_plot)
        all_x_labels.extend(x_labels)

    # Only show these ticks/labels on the x axis
    show_labels = ['0', '100', '1000', '10000', '100000']
    show_x_plot = []
    for label in show_labels:
        # Find the first matching x value for this label
        for x, lab in zip(all_x_plot, all_x_labels):
            if lab == label:
                show_x_plot.append(x)
                break

    for xi in show_x_plot:
        ax.axvline(x=xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xticks(show_x_plot)
    ax.set_xticklabels(show_labels)
    ax.set_xlabel(r'$\log_{10} r$ (pseudo-$\log_{10} r$ | $r=0$)', labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    apply_custom_style(ax)
    ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, save_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # Save as EPS
    eps_save_path = os.path.splitext(save_path)[0] + ".eps"
    plt.savefig(eps_save_path, format='eps', bbox_inches='tight')
    plt.show()

#%% --- Ask user which chart to plot ---
if __name__ == "__main__":
    chart_map = {
        "WA": plot_WA_chart,
        "WR": plot_WR_chart,
        "BR": plot_BR_chart,
    }
    print("Which chart do you want to plot? (WA/WR/BR)")
    chart_choice = input("Enter chart name: ").strip().upper()
    if chart_choice in chart_map:
        chart_map[chart_choice](chart_configs[chart_choice])
    else:
        print("Invalid choice. Please enter WA, WR, or BR.")


### --- TABLE 2 - ACCURACY OF WATERMARKED NEURAL NETWORKS
#%%
data = [
    [96.48, 96.46, 96.47, 96.47],
    [96.58, 96.57, 96.57, 96.58],
    [93.29, 93.30, 93.28, 93.28],
    [95.04, 95.03, 95.07, 95.03],
    [96.21, 96.20, 96.23, 96.21]
]

# X-axis values: 0, 4, 6, 8
x = [0, 4, 6, 8]
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

fig, ax = plt.subplots(figsize=(4.5, 3))
for i, y in enumerate(data):
    # Plot line without label
    ax.plot(x, y, color=colors[i], linewidth=0.8)
    # Plot markers with label for legend only
    ax.plot(x, y, color=colors[i], marker='o', linestyle='None', markersize=2.5, label=f'NN{i+1}')


# Add vertical dashed lines
for xi in range(0, 9, 2):
    ax.axvline(x=xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

ax.set(xlim=(0, 8), ylim=(93, 97), xticks=range(0, 9, 2),
       xlabel='Epoch', ylabel='Accuracy (%)')

ax.set_yticks(np.arange(93, 97.5, 1))  # steps of 1 from 93 to 97


apply_custom_style(ax)  # Reuse your existing style function
ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Save plot
save_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'chart_accuracy_watermarked')
plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight')
plt.savefig(save_path + '.eps', format='eps', bbox_inches='tight')
plt.show()

