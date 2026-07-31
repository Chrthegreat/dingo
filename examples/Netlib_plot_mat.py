import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# THIS FILE PLOTS THE biology_benchmarks.csv

filename = 'Netlib_mat_benchmark.csv' 
df = pd.read_csv(filename, skipinitialspace=True)

df.columns = [col.strip() for col in df.columns]
df['Model'] = df['Model'].astype(str).str.strip()
df['Method'] = df['Method'].astype(str).str.strip()

df = df.sort_values(by="Dimension")

methods = df['Method'].unique()
cmap = plt.get_cmap('tab10')
colors = {method: cmap(i % 10) for i, method in enumerate(methods)}

unique_dims = df["Dimension"].unique()
model_names_for_ticks = []
for d in unique_dims:
    model_name = df[df["Dimension"] == d]["Model"].iloc[0]
    model_names_for_ticks.append(f"{model_name}\n(d={d})")

# --- Font Size Configurations ---
TITLE_SIZE = 12
SUBTITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 6.4
LEGEND_SIZE = 8

fig, ax = plt.subplots(figsize=(8, 5))

for method in methods:
    subset = df[df['Method'] == method]
    ax.plot(
        subset['Dimension'],
        subset['Time_sec'],
        marker='o',
        linewidth=1.5,
        color=colors[method],
        label=method
    )

ax.set_title('Rounding Time', fontsize=SUBTITLE_SIZE)
ax.set_ylabel('Time to Round (seconds) [Log]', fontsize=LABEL_SIZE)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(unique_dims)
ax.set_xticklabels(
    model_names_for_ticks,
    rotation=45,
    ha='right',
    fontsize=TICK_SIZE
)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.grid(True, linestyle='--', alpha=0.7)

ax.legend(loc='upper left', fontsize=LEGEND_SIZE)

plt.tight_layout()
plt.show()