from globals import *
from predict_full_seq5 import predict_full_sequence
from processing import *
import random

num_iter = 200
saving = True

torch.manual_seed(409)
random.seed(409)

if saving:

    for i in range(num_iter):


        # Load data
        n_ind_adm_start_tmp = random.randint(0, 399)
        n_ind_adm_end_tmp = n_ind_adm_start_tmp + 1

        print(n_ind_adm_start_tmp)
        print(n_ind_adm_end_tmp)


        random_file = random.randint(0, num_files - 1)


        with open(parameters_dir + "parameter_" + str(random_file)) as f:
            admixture_proportion, num_generations, *_ = f.readlines()
            admixture_proportion = float(admixture_proportion)
            num_generations = int(num_generations)

        X, positions = convert_panel(panel_dir + "panel_" + str(random_file), n_ind_adm_start=n_ind_adm_start_tmp, n_ind_adm_end=n_ind_adm_end_tmp)
        X = torch.tensor(X).to(device).squeeze(0) # len_seq
        positions = torch.tensor(positions).to(device) # len_seq

        print(X.shape)
        print(positions.shape)

        y = convert_split(split_dir + "split_" + str(random_file), positions, n_ind_adm_start=n_ind_adm_start_tmp, n_ind_adm_end=n_ind_adm_end_tmp)
        y = torch.tensor(y).to(device) # 2 * n_ind_adm, len_seq
        y = y[0] + y[1] # len_seq

        print(y.shape)

        refA, refB, _ = convert_panel_template(panel_template_dir + "panel_template_" + str(random_file)) 

        refA = torch.tensor(refA).to(device) #num_files, n_ind_pan, input_size
        refB = torch.tensor(refB).to(device) #num_files, n_ind_pan, input_size
        refs = torch.zeros((num_classes, n_ind_pan_model, refA.shape[-1])).to(device)
        refs[0] = refA[:2 * n_ind_pan // 6 * 2:2] + refA[1:2 * n_ind_pan // 6 * 2:2]
        refs[2] = refB[:2 * n_ind_pan // 6 * 2:2] + refB[1:2 * n_ind_pan // 6 * 2:2]
        refs[1] = refA[-(n_ind_pan // 6 * 2):] + refB[-(n_ind_pan // 6 * 2):]
        refs[1] = -1 ####################!!!!!!!!!!!!!!!!!!!!
        refs = refs.reshape(n_ind_max, -1)

        labels = torch.arange(num_classes).to(device).unsqueeze(-1).unsqueeze(-1)
        labels = labels.repeat(1, n_ind_pan_model, refs.shape[-1])

        labels = F.one_hot(labels.long(), num_classes=num_classes)
        labels[1] = 0 ######################!!!!!!!!!!!!!!!!!!!
        labels = labels.reshape(n_ind_max, -1, num_classes)