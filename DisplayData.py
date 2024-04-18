from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from adjustText import adjust_text

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

##set root_dir to the correct path to your dataset folder
root_dir = './data/val/data/'
agent_results_dir = './data/results/AGENT'
av_results_dir = './data/results/AV'

afl = ArgoverseForecastingLoader(root_dir)

#set csv_filename to desired scenario
csv_filename = '14986'
(x_agent_start, y_agent_start, x_av_start, y_av_start) = viz_sequence(afl.get(os.path.join(root_dir, csv_filename+".csv")).seq_df, show=False)
with open(os.path.join(agent_results_dir, csv_filename + ".csv"), 'r') as f:
    reader = csv.reader(f)
    agent_data_temp = list(reader)
with open(os.path.join(av_results_dir, csv_filename + ".csv"), 'r') as f:
    reader = csv.reader(f)
    av_data_temp = list(reader)
x_agent_end_data = []
y_agent_end_data = []
x_av_end_data = []
y_av_end_data = []

x_agent = []
y_agent = []
x_av = []
y_av = []
x_agent.append(x_agent_start)
y_agent.append(y_agent_start)
x_av.append(x_av_start)
y_av.append(y_av_start)
first_traj = True
i = 3
while i < len(agent_data_temp[0])-1:
    if agent_data_temp[0][i] == '|':
        i = i + 1
        x_agent_end_data.append(x_agent[-1])
        y_agent_end_data.append(y_agent[-1])
        x_av_end_data.append(x_av[-1])
        y_av_end_data.append(y_av[-1])
        if first_traj:
            plt.plot(x_agent,y_agent, "-", color = '#1a80bb',linewidth=2, markersize=3, label = "Vehicle 1 Predicted Trajectory")
            plt.plot(x_av, y_av, "-", color = '#ea801c',linewidth=2, markersize=3, label = "Vehicle 2 Predicted Trajectory")
            first_traj = False
        else:
            plt.plot(x_agent,y_agent, "-", color = '#1a80bb',linewidth=2, markersize=3)
            plt.plot(x_av, y_av, "-", color = '#ea801c',linewidth=2, markersize=3)
        x_agent = []
        y_agent = []
        x_av = []
        y_av = []
        x_agent.append(x_agent_start)
        y_agent.append(y_agent_start)
        x_av.append(x_av_start)
        y_av.append(y_av_start)
    else:
        x_agent.append(float(agent_data_temp[0][i]))
        y_agent.append(float(agent_data_temp[0][i+1]))
        x_av.append(float(av_data_temp[0][i]))
        y_av.append(float(av_data_temp[0][i+1]))
        i = i + 2
    
plt.scatter(x_agent_end_data, y_agent_end_data, color='#1a80bb', linewidths=3,marker=".")
plt.scatter(x_av_end_data, y_av_end_data, color='#ea801c', linewidths=3,marker=".")

agent_texts = [plt.text(x_agent_end_data[j],y_agent_end_data[j],f"$s_{j}$", fontsize=16) for j in range(6)]
av_texts = agent_texts + [plt.text(x_av_end_data[j],y_av_end_data[j],f"$s_{j}$", fontsize=16) for j in range(6)]
adjust_text(av_texts, avoid_self=True, min_arrow_len = 5, force_explode=(2,2), force_static=(1,1),force_text=(1,1), only_move={'explode':'xy', 'pull':'xy', 'static':'xy', 'text':'xy'},arrowprops=dict(arrowstyle='-|>', mutation_scale=10, color='black'),zorder=100)


with open(os.path.join(agent_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    agent_data_temp = list(reader)
with open(os.path.join(av_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    av_data_temp = list(reader)
row_labels = ["$s_0$","$s_1$","$s_2$","$s_3$","$s_4$","$s_5$"]
col_labels = ["$\Pr(x_1,y_1)$","$\Pr(x_2,y_2)$",]
table_vals = np.zeros((6,2))
i = 2
for j in range(6):
    table_vals[j][0] = "{:.3e}".format(float(agent_data_temp[0][i]))
    table_vals[j][1] = "{:.3e}".format(float(av_data_temp[0][i]))
    i = i + 2
my_table = plt.table(cellText=table_vals,
                     colWidths = [0.2] * 6,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='lower right',
                     fontsize=16)
my_table.scale(1,2)
my_table.auto_set_font_size(False)
my_table.set_fontsize(16)

plt.legend(fontsize=16, loc='upper left')
# Edit the xlim and ylim to scale as desired
plt.xlim(1740,1865)
plt.ylim(370,455)
plt.axis("off")
ax = plt.gca()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(os.path.join('./data/results/', csv_filename[:-4]+'.png')) 
plt.show()