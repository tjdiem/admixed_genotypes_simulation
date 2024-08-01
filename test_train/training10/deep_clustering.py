from globals import *
from processing import *
from training_functions import preprocess_batch
from kmodels2 import KNet4
import sys

random_file = int(sys.argv[1])

torch.manual_seed(409)
random.seed(409)

with open(parameters_dir + "parameter_" + str(random_file)) as f:
    admixture_proportion, num_generations, *_ = f.readlines()
    admixture_proportion = float(admixture_proportion)
    num_generations = int(num_generations)
    
len_seq = 950

print(random_file)
X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device)[:49,:len_seq] # n_ind_adm, input_size
positions = torch.tensor(positions).to(device)[:len_seq]

print(X.sum(dim=0))
# print(X.shape)

y = convert_split(split_dir + "split_" + str(random_file), positions)
y = torch.tensor(y) # 2 * n_ind_adm, input_size
# y = y[:49*2:2, :len_seq] + y[1:49*2:2, :len_seq]
y = (y[::2] + y[1::2])[:49,:len_seq] # unphase ancestry labels # same shape as X


model = KNet4()
model = model.to(device)
model.load_state_dict(torch.load("model_1_pos_time.pth", map_location=torch.device(device)))
model.eval()

num_bp = 50_000_000
recombination_map = lambda x: x / num_bp  # this assumes recombination rate is constant along chromosome, for chromosome of 50M base pairs
population_size = 10_000 # from looking at demography file. Should we vary this parameter?





t1 = time.time()
y_pred = model.predict_cluster3(X, positions, recombination_map=recombination_map, batch_size=16, num_generations=num_generations, admixture_proportion=None)
print(time.time() - t1)
y_pred = y_pred.argmax(dim=-1)
y = y.to(device)
for i in range(3):
    print((y_pred == i).sum().item())
    print((y == i).sum().item())
    print()

acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
print(acc)
