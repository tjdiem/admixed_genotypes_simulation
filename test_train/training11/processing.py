from globals import *
import os
import subprocess
import random
import tarfile
import gc

random.seed(random_seed)

def GetCudaMemory():
    allocated_tensors = []
    total_memory = 0
    
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            allocated_tensors.append((obj, obj.numel() * obj.element_size()))
            total_memory += obj.numel() * obj.element_size()

    print(f"Total allocated CUDA memory: {total_memory / (1024 ** 2):.2f} MB")
    for tensor, size in allocated_tensors:
        print(f"{tensor.shape}: {size / (1024 ** 2):.2f} MB")

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

def convert_panel(panel_file):

    end_index = None if input_size_processing == "all" else input_size_processing*input_size_step

    try:
        with open(panel_file, "r") as f:
            lines = f.readlines()[:end_index:input_size_step]
    except FileNotFoundError:
        with tarfile.open(panel_file + ".tar.xz", 'r:xz') as tar:
            file_obj = tar.extractfile(tar.getnames()[0]) #panel_file.split("/")[-1])
            assert file_obj is not None
            lines = file_obj.readlines()[:end_index:input_size_step]

        lines = [line.decode("utf-8").strip() for line in lines]


    lines = [line.split() for line in lines]

    positions = [int(line[1]) for line in lines]

    admixed_individual = [[int(line[i]) for line in lines] for i in range(8 + n_ind_adm_start * 2, 8 + n_ind_adm_end * 2, 2)]
    
    return admixed_individual, positions

def convert_panel2(panel_file, input_size_start, input_size_step, input_size_end, ind_adm):

    end_index = None if input_size_processing == "all" else input_size_processing*input_size_step

    lines = []
    with tarfile.open(panel_file + ".tar.xz", 'r') as tar:
        member = tar.getmember(tar.getnames()[0]) #panel_file.split("/")[-1])
        with tar.extractfile(member) as file:

            for i, line in enumerate(file, start=0):
                if i < input_size_start:
                    continue
                if i >= input_size_end:
                    break
                if (i % input_size_step) == (input_size_start % input_size_step):
                    lines.append(line.decode('utf-8').strip())

    lines = [line.split() for line in lines]

    positions = [int(line[1]) for line in lines]

    admixed_individual = [[int(line[8 + 2 * i]) for line in lines] for i in ind_adm]

    # admixed_individual = [[int(line[i]) for line in lines] for i in range(8 + n_ind_adm_start * 2, 8 + n_ind_adm_end * 2, 2)]
    
    return admixed_individual, positions


def convert_panel_template(panel_template_file):

    end_index = None if input_size_processing == "all" else input_size_processing*input_size_step+1

    try:
        with open(panel_template_file, "r") as f:
            lines = f.readlines()[1:end_index:input_size_step]
    except FileNotFoundError:
        with tarfile.open(panel_template_file + ".tar.xz", 'r:xz') as tar:
            file_obj = tar.extractfile(tar.getnames()[0]) #panel_template_file.split("/")[-1])
            assert file_obj is not None
            lines = file_obj.readlines()[1:end_index:input_size_step]

        lines = [line.decode("utf-8").strip() for line in lines]

    lines = [line.split() for line in lines]

    positions = [int(line[1]) for line in lines]

    popA = [line[-2][:n_ind_pan] for line in lines]
    popB = [line[-1][:n_ind_pan] for line in lines]

    popA = [[int(popA[i][j]) for i in range(len(lines))] for j in range(n_ind_pan)]
    popB = [[int(popB[i][j]) for i in range(len(lines))] for j in range(n_ind_pan)]

    return popA, popB, positions

def convert_split(split_file, positions):

    with open(split_file, "r") as f:
        splits = f.readlines()[:-1][2*n_ind_adm_start:2*n_ind_adm_end]

    splits = [[-1.0] + [float(l) for l in line.split("\t")] + [1.0] if not line.isspace() else [-1.0, 1.0] for line in splits]
    assert len(splits) == 2 * (n_ind_adm_end - n_ind_adm_start)

    y = []
    for split in splits:
        y_ind = []
        idx = 0
        for position in positions:
            rel_pos = position / num_bp
            while rel_pos > split[idx]:
                idx += 1

            assert split[idx - 1] < rel_pos <= split[idx]
            y_ind.append((idx + 1) % 2)

        y.append(y_ind)

    return y

def convert_parameter(parameter_file):

    with open(parameter_file, "r") as f:
        lines = f.readlines()

    return [float(line) for line in lines]

