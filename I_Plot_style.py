def apply_custom_style(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, fontname='Times New Roman')
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, fontname='Times New Roman')

    ax.tick_params(axis='x', which='major', length=5, width=1, direction='in', color='gray', labelsize=10)
    ax.tick_params(axis='y', which='major', length=5, width=1, direction='in', color='gray', labelsize=10)

    # Set font for tick labels (safer than set_xticklabels/set_yticklabels)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Grid only on y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    # Set all spines to gray
    for spine in ax.spines.values():
        spine.set_color('gray')
