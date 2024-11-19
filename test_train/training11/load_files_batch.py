from processing import * 
import random
import torch
import time
import concurrent.futures

file = int(sys.argv[1])

start_time = time.time()

start = random.randint(0, 10000)

ind_adm = list(range(400))
random.shuffle(ind_adm)
ind_adm = ind_adm[:49]

# args = [("../../data_simulated/panels14/panel_" + str(file), start,1,start+1000,ind_adm) for file in files]
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     results = list(executor.map(lambda p: convert_panel2(*p), args))

x_ = []
pos_ = []

x, pos = convert_panel2("../../data_simulated/panels14/panel_" + str(file), start,1,start+1000,ind_adm)
x_.append(x)
pos_.append(pos)

x = torch.tensor(x)
print(x.shape)




