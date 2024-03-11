from globals import *
import os
import subprocess
import random

random.seed(random_seed)

def GetMemory():
    if os.name == 'posix':
        mem_info = subprocess.check_output(['free','-b']).decode().split()
        total_memory = int(mem_info[7]) - int(mem_info[8]) 
        total_memory *= 10**-9
        
    elif os.name == 'nt':
        mem_info = subprocess.check_output(['wmic','OS','get','FreePhysicalMemory']).decode().split()
        total_memory = int(mem_info[1]) * 1024 * 10**-9
        
    print(f"Available memory: {total_memory:0.2f} GB")
    
def GetTime():
    seconds = int(time.time() - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")

def sample_point(starting_pos, pos, split_points):
        
    while pos > split_points[starting_pos]:
        starting_pos += 1
    
    #assert (split_points[starting_pos - 1] < pos <= split_points[starting_pos]) or (starting_pos == 0 and pos < split_points[starting_pos]), f"{split_points[starting_pos - 1]}, {pos}, {split_points[starting_pos]}"
    return starting_pos

def convert_output_file(panel_file, phase_file):

    with open(panel_file, "r") as f:
        panel_lines = f.readlines()[:input_size_step*input_size_processing]

    with open(phase_file, "r") as f:
        phase_split_lines = f.readlines()[:n_embd_processing*2]

    split_lines = [[float(l) for l in line.split("\t")] + [1.0] for line in phase_split_lines[::2]]
    phase_lines = [[int(l) for l in line.split("\t")] for line in phase_split_lines[1::2]]

    split_positions = [0 for _ in range(len(split_lines))]

    X = []
    y = []
    pos = 0.0
    for i, line in enumerate(panel_lines):

        line_info = line.split("\t")
        pos += float(line_info[6])

        if i % input_size_step != 0:
            continue

        # print(pos)
        #y.append(int(line_info[2])/(int(line_info[2]) + int(line_info[4])))
        X.append([int(i) for i in line_info[7::2][:n_embd_processing]] + [pos])

        # Find ancestry (y value) of each sample based on split point
        split_positions = [sample_point(split_positions[i], pos, split_lines[i]) for i in range(n_embd_processing)]
        y.append([phase_lines[i][split_positions[i] - 1] if split_positions[i] != 0 else 0 for i in range(n_embd_processing)])
            

    
    return (X, y)

