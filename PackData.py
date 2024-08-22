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
    temp = f"{agent_data_temp[0][0]},{agent_data_temp[0][1]},{av_data_temp[0][0]},{av_data_temp[0][1]},|,"
    i = 3
    while i < len(agent_data_temp[0])-1:
        if agent_data_temp[0][i] == '|':
            i = i + 1
        else:
            j = 3
            k = i
            while j < len(av_data_temp[0])-1:
                if av_data_temp[0][j] == '|':
                    j = j + 1
                    k = i
                    temp = temp + "|,"
                else:
                    temp = temp + f"{agent_data_temp[0][k]},{agent_data_temp[0][k+1]},{av_data_temp[0][j]},{av_data_temp[0][j+1]},"
                    j = j + 2
                    k = k + 2
            i = i + (2*30)
    f.write(temp)

with open(os.path.join(agent_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    agent_data_temp = list(reader)
with open(os.path.join(av_results_dir, csv_filename + "_prob.csv"), 'r') as f:
    reader = csv.reader(f)
    av_data_temp = list(reader)

with open(os.path.join('./data/results/', csv_filename+'_trace_prob.csv'), 'w') as f:
    temp = f"{float(agent_data_temp[0][0])*float(av_data_temp[0][0])},|,"
    i = 2
    while i < len(agent_data_temp[0])-1:
        j = 2
        while j < len(av_data_temp[0])-1:
            prob = float(agent_data_temp[0][i])*float(av_data_temp[0][j])
            for k in range(30):
                temp = temp + str(prob) + ","
            temp = temp + "|,"
            j = j + 2
        i = i + 2
    f.write(temp)