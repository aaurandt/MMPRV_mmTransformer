import csv
import os

agent_results_dir = './data/results/AGENT'
av_results_dir = './data/results/AV'

# set csv_filename to desired scenario
csv_filename = '14986'
with open(os.path.join(agent_results_dir, csv_filename + ".csv"), 'r') as f:
    reader = csv.reader(f)
    agent_data_temp = list(reader)
with open(os.path.join(av_results_dir, csv_filename + ".csv"), 'r') as f:
    reader = csv.reader(f)
    av_data_temp = list(reader)

with open(os.path.join('./data/results/', csv_filename+'_trace.csv'), 'w') as f:
    f.write("# x_1, y_1, x_2, y_2\n")
    temp = ""
    i = 0
    while i < len(agent_data_temp[0])-1:
        if agent_data_temp[0][i] == '|':
            i = i + 1
            temp = temp + "|,"
        else:
            temp = temp + f"{agent_data_temp[0][i]},{agent_data_temp[0][i+1]},{av_data_temp[0][i]},{av_data_temp[0][i+1]},"
            i = i + 2
    f.write(temp)

with open(os.path.join(agent_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    agent_data_temp = list(reader)
with open(os.path.join(av_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    av_data_temp = list(reader)

with open(os.path.join('./data/results/', csv_filename+'_trace_prob.csv'), 'w') as f:
    i = 0
    while i < len(agent_data_temp[0])-1:
        prob = float(agent_data_temp[0][i])*float(av_data_temp[0][i])
        if i == 0:
            temp = str(prob) + ",|,"
        else:
            for j in range(30):
                temp = temp + str(prob) + ","
            temp = temp + "|,"
        i = i + 2
    f.write(temp)