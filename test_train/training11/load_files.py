from processing import * 
import random
import torch
import time
import concurrent.futures

start_time = time.time()
def wrapper(args):
    return convert_panel2(*args)

def get_batch(files):
    start = random.randint(0, 10000)

    ind_adm = list(range(400))
    random.shuffle(ind_adm)
    ind_adm = ind_adm[:49]


    args = [("../../data_simulated/panels14/panel_" + str(file), start,1,start+1000,ind_adm) for file in files]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(wrapper, args))

    # x_ = []
    # pos_ = []
    # for file in files:
    #     x, pos = convert_panel2("../../data_simulated/panels14/panel_" + str(file), start,1,start+1000,ind_adm)
    #     x_.append(x)
    #     pos_.append(pos)

    return None, None # torch.tensor(x), torch.tensor(pos)

all_files = list(range(367))
for epoch in range(100):
    for istart in range(0, 367, 16):
        iend = min(368, istart + 16)
        x, pos = get_batch(all_files[istart: iend])
        print(time.time() - start_time)

