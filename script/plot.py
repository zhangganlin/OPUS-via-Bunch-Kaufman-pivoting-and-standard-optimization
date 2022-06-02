import matplotlib.pyplot as plt
import yaml
import numpy as np
from matplotlib.lines import Line2D  
import csv

def plot_total(path, save = False, save_dir = None):
    with open(path, 'r') as f:
        result = yaml.full_load(f)
        cost_for_each_step_total = []
    cost_for_each_step_total.append(result['step1to4'])
    cost_for_each_step_total.append(sum(result['step5_time']))
    cost_for_each_step_total.append(sum(result['step6a_time']))
    cost_for_each_step_total.append(sum(result['step6b_time']))
    cost_for_each_step_total.append(sum(result['step7_time']))
    cost_for_each_step_total.append(sum(result['step8_time']))
    cost_for_each_step_total.append(sum(result['step9_time']))
    cost_for_each_step_total.append(sum(result['step10_time']))
    cost_for_each_step_total.append(sum(result['step11_time']))

    cost_for_each_step_total_name = ["s1-s4","s5", "s6a", "s6b","s7", "s8", "s9", "s10", "s11"]
    fig, ax = plt.subplots()
    ax.bar(range(len(cost_for_each_step_total)),cost_for_each_step_total,tick_label=cost_for_each_step_total_name, color='black')
    ax.set_xlabel('Steps (1 - 11)')
    ax.set_ylabel('[cycles]', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Running time of OPUS on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n', loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if save:
        fig.tight_layout()
        fig.savefig(save_dir, format = 'eps')

def plot_two_total(path1, path2, save, save_dir):
    with open(path1, 'r') as f:
        result = yaml.full_load(f)
    cost_for_each_step_total1 = []
    cost_for_each_step_total1.append(result['step1to4'])
    cost_for_each_step_total1.append(sum(result['step5_time']))
    cost_for_each_step_total1.append(sum(result['step6a_time']))
    cost_for_each_step_total1.append(sum(result['step6b_time']))
    cost_for_each_step_total1.append(sum(result['step7_time']))
    cost_for_each_step_total1.append(sum(result['step8_time']))
    cost_for_each_step_total1.append(sum(result['step9_time']))
    cost_for_each_step_total1.append(sum(result['step10_time']))
    cost_for_each_step_total1.append(sum(result['step11_time']))

    with open(path2, 'r') as f:
        result = yaml.full_load(f)
    cost_for_each_step_total2 = []
    cost_for_each_step_total2.append(result['step1to4'])
    cost_for_each_step_total2.append(sum(result['step5_time']))
    cost_for_each_step_total2.append(sum(result['step6a_time']))
    cost_for_each_step_total2.append(sum(result['step6b_time']))
    cost_for_each_step_total2.append(sum(result['step7_time']))
    cost_for_each_step_total2.append(sum(result['step8_time']))
    cost_for_each_step_total2.append(sum(result['step9_time']))
    cost_for_each_step_total2.append(sum(result['step10_time']))
    cost_for_each_step_total2.append(sum(result['step11_time']))

    cost_for_each_step_total_name = ["s1-s4","s5", "s6a", "s6b","s7", "s8", "s9", "s10", "s11"]
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps (1 - 11)')
    ax.set_ylabel('[cycles]', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Running time of OPUS on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n', loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    width = 0.35  # the width of the bars
    x = np.arange(len(cost_for_each_step_total1))
    rects1 = ax.bar(x - width/2, cost_for_each_step_total1, width, label='before optimization', color='black')
    rects2 = ax.bar(x + width/2, cost_for_each_step_total2, width, label='after optimization', color='brown')
    ax.set_xticks(x, cost_for_each_step_total_name)
    ax.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    if save:
        fig.savefig(save_dir, dpi = 600, format = 'eps')

def plot_step_performance(path, step_name, save, save_dir):
    with open(path, 'r') as f:
        result = yaml.full_load(f)
    plt.figure(figsize=(20,14), dpi= 160)
    run_time = result['step' + step_name + '_time']
    flops = result['step' + step_name + '_flop']
    performance = [w/t for t, w in zip(run_time, flops)]
    fig, ax = plt.subplots()
    x_list = np.arange(0, len(performance), 1)
    
    ax.set_xlabel('iterations')
    ax.set_ylabel('[flops/cycle]', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Performance of OPUS (step{}) on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n'.format(step_name), loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(x_list, performance, marker = "o", color = 'brown', markersize= 3)
    if save:
        fig.savefig(save_dir, dpi = 600, format = 'eps')

def plot_two_step_performance(path1, path2, step_name, save, save_dir):
    with open(path1, 'r') as f:
        result = yaml.full_load(f)
    plt.figure(figsize=(20,14), dpi= 160)
    run_time = result['step' + step_name + '_time']
    flops = result['step' + step_name + '_flop']
    performance1 = [w/t for t, w in zip(run_time, flops)]

    with open(path2, 'r') as f:
        result = yaml.full_load(f)
    plt.figure(figsize=(20,14), dpi= 160)
    run_time = result['step' + step_name + '_time']
    flops = result['step' + step_name + '_flop']
    performance2 = [w/t for t, w in zip(run_time, flops)]


    fig, ax = plt.subplots()
    x_list = np.arange(0, len(performance1), 1)
    
    ax.set_xlabel('iterations')
    ax.set_ylabel('[flops/cycle]', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Performance of OPUS (step{}) on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n'.format(step_name), loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(x_list, performance1, marker = "o", color = 'black', markersize= 3, label = 'before optimization')
    ax.plot(x_list, performance2, marker = "o", color = 'brown', markersize= 3, label = 'after optimization')
    ax.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    if save:
        fig.savefig(save_dir, dpi = 600, format = 'eps')
    
def plot_step_performances(path, name, x_axis_name, save, save_dir):
    with open(path, 'r') as f:
        result = yaml.full_load(f)
    result['evaluate_func_cycles'] = (np.array(result['evaluate_func_cycles']).T).tolist()
    result['evaluate_func_flops'] = (np.array(result['evaluate_func_flops']).T).tolist()
    plt.figure(figsize=(20,14), dpi= 160)
    fig, ax = plt.subplots()
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel('[flops/cycle]', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Performance of {} function on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n'.format(name), loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    handles = []
    
    for i in range(len(result['evaluate_func_name'])):
        cycles = result['evaluate_func_cycles'][i]
        flops = result['evaluate_func_flops'][i]
        performance = [w/t for t, w in zip(cycles, flops)]
        x_list = np.arange(result['n_start'], result['n_end'], result['n_gap'])
        if i == len(result['evaluate_func_name']) - 1:
            p, = ax.plot(x_list, performance, marker = "o", color = 'brown', linewidth = 2, markersize= 4, label = result['evaluate_func_name'][i])
        else:
            p, = ax.plot(x_list, performance, marker = "o", markersize= 3, label = result['evaluate_func_name'][i])
        handles.append(p)
    handles.reverse()
    ax.legend(handles=handles, title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    if save:
        fig.savefig(save_dir, dpi = 600, format = 'eps')

def plot_blocksize_speedup(path, save, save_dir):
    path = '../output/blocksize_speedup.txt'
    with open(path, newline='') as csvfile:
        result = []
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print(spamreader)
        for row in spamreader:
            result.append([(float(string)) for string in row])
    result = np.array(result).T
    plt.figure(figsize=(20,14), dpi= 160)
    fig, ax = plt.subplots()
    ax.set_xlabel('block size(double)')
    ax.set_ylabel('speedup', loc = 'top', rotation="horizontal")
    ax.grid(axis="y", color='white')
    ax.set_facecolor(color='gainsboro')
    ax.set_title('Speedup w.r.t block size on Intel i7-7560 CPU, 2.40GHz\nCompiler: GCC 9.4.0\nFlags:-march=native\n', loc='left', fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(result[0], result[1], color = 'black', marker = 'o', markersize = 3)
    if save:
        fig.savefig(save_dir, dpi = 600, format = 'eps')
