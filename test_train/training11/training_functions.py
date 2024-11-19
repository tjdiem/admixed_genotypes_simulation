from globals import *


def preprocess_batch(X_batch, y_batch, refs_batch, prob_args=None):

    n_ind_adm = n_ind_adm_end - n_ind_adm_start

    random_items = torch.rand((10,)).tolist()

    # X_batch, y_batch (batch, n_ind_adm, input_size)
    # refs_batch   (batch, num_classes, n_ind_pan, input_size)

    if prob_args is None:
        prob0 = 0.5
        prob1 = 1.0
        prob2a = 0.6
        prob2b = 0.2
        prob2c = 0.2
        prob2d = 0
        prob2e = 0
        prob3 = 1.0

    else:
        prob_args_dict, split = prob_args
        prob0 = prob_args_dict["prob0"][split]
        prob1 = prob_args_dict["prob1"][split]
        prob2a = prob_args_dict["prob2a"][split]
        prob2b = prob_args_dict["prob2b"][split]
        prob2c = prob_args_dict["prob2c"][split]
        prob2d = prob_args_dict["prob2d"][split]
        prob2e = prob_args_dict["prob2e"][split]
        prob3 = prob_args_dict["prob3"][split]

    random_tensor = torch.randint(0, n_ind_adm, (X_batch.shape[0],))

    X_batch_out = X_batch[torch.arange(X_batch.shape[0]).long(), random_tensor]
    y_batch_out = y_batch[torch.arange(X_batch.shape[0]).long(), random_tensor, input_size // 2]

    if random_items[0] < prob0:

        # create refs_batch (out) and labels_batch from refs_batch
        labels_batch = torch.arange(num_classes).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        labels_batch = labels_batch.repeat(refs_batch.shape[0], 1, n_ind_pan_model, input_size)

        refs_batch = refs_batch.reshape(refs_batch.shape[0], n_ind_max, input_size)
        labels_batch = labels_batch.reshape(labels_batch.shape[0], n_ind_max, input_size)
        labels_batch = F.one_hot(labels_batch.long(), num_classes=num_classes)
        labels_batch[refs_batch == -1] = -1

        if random_items[1] < prob1:
            # block out arbitrary number of reference panels 
            rand_arange = torch.stack([torch.randperm(n_ind_max) for _ in range(refs_batch.shape[0])])
            rand_int = torch.randint(0, n_ind_max, (refs_batch.shape[0], 1))
            idx_blocked = (rand_arange < rand_int)#.unsqueeze(-1).repeat(1, 1, input_size)
            refs_batch[idx_blocked] = -1
            labels_batch[idx_blocked] = -1
        else:
            # block out arbitrary number of refrence panels from each class
            rand_arange = torch.stack([torch.randperm(n_ind_max) for _ in range(refs_batch.shape[0])])
            rand_int = torch.cat([torch.randint(0, n_ind_max, (refs_batch.shape[0], 1)).repeat(1, n_ind_pan_model) for _ in range(num_classes)], dim=1)
            idx_blocked = (rand_arange < rand_int)#.unsqueeze(-1).repeat(1, 1, input_size)
            refs_batch[idx_blocked] = -1
            labels_batch[idx_blocked] = -1

        if random_items[2] < prob2a:
            # block out arbitrary SNPs for each reference panel
            rand_float = torch.rand(refs_batch.shape[0], n_ind_max)
            rand_int = torch.randint(1, input_size, (refs_batch.shape[0], n_ind_max, 1))
            rand_arange = torch.arange(input_size).unsqueeze(0).unsqueeze(0).repeat(refs_batch.shape[0], n_ind_max, 1)
            
            rand_arange[rand_float < 1/3] = input_size
            rand_arange[rand_float > 2/3] = input_size - 1 - rand_arange[rand_float > 2/3]
            
            idx_blocked = (rand_arange < rand_int)
            refs_batch[idx_blocked] = -1
            labels_batch[idx_blocked] = -1

        elif random_items[2] < prob2a + prob2b:
            # block out first half (rounding down) of each reference panel
            refs_batch[:,:,:input_size // 2] = -1
            labels_batch[:,:,:input_size // 2] = -1

        elif random_items[2] < prob2a + prob2b + prob2c:
            # block out second half (rounding up) of each reference panel
            refs_batch[:,:,input_size // 2 + 1:] = -1
            labels_batch[:,:,input_size // 2 + 1:] = -1

        elif random_items[2] < prob2a + prob2b + prob2c + prob2d:
            rand_int = torch.randint(1, input_size // 2 + 1, (1,)).item()
            refs_batch[:,:,:rand_int] = -1
            labels_batch[:,:,:rand_int] = -1

        elif random_items[2] < prob2a + prob2b + prob2c + prob2d + prob2e:
            rand_int = torch.randint(input_size // 2 + 1, input_size, (1,)).item()
            refs_batch[:,:,rand_int:] = -1
            labels_batch[:,:,rand_int:] = -1

        else:
            # don't block any SNPs
            pass


    else:
        # create refs_batch (out) and labels_batch from X_batch and y_batch
        arange = torch.arange(n_ind_adm)
        rand_perms = [torch.randperm(n_ind_adm - 1)[:n_ind_max] for _ in range(X_batch.shape[0])]
        refs_batch = torch.stack([X_batch.clone().to(device)[i, arange != random_tensor[i]][rand_perms[i]] for i in range(X_batch.shape[0])])
        labels_batch = torch.stack([y_batch.clone().to(device)[i, arange != random_tensor[i]][rand_perms[i]] for i in range(X_batch.shape[0])])
        labels_batch = F.one_hot(labels_batch.abs().long(), num_classes=num_classes)
        labels_batch[refs_batch == -1] = -1

        #block out arbitrary number of reference panels
        rand_arange = torch.stack([torch.randperm(n_ind_max) for _ in range(refs_batch.shape[0])])
        rand_int = torch.cat([torch.randint(0, n_ind_max, (refs_batch.shape[0], 1)).repeat(1, n_ind_pan_model) for _ in range(num_classes)], dim=1)
        idx_blocked = (rand_arange < rand_int)#.unsqueeze(-1).repeat(1, 1, input_size)
        refs_batch[idx_blocked] = -1
        labels_batch[idx_blocked] = -1

        # block out first or second half?

    if random_items[3] < prob3:
        # block out reference panels of mixed ancestry
        idx_blocked = labels_batch[...,1].bool()
        refs_batch[idx_blocked] = -1
        labels_batch[idx_blocked] = -1

    ##### keeping this in when training w/out probs seems to help
    # if train_with_probs:
    #     # We assume batch_size == 1
    #     labels_batch = labels_batch.float()
    #     num_probs = torch.randint(0, max_num_probs + 1, (1,)).item()
    #     valid_prob_indices = (labels_batch == 0).nonzero()
    #     rand_perm = torch.randperm(valid_prob_indices.shape[0])[:min(num_probs, valid_prob_indices.shape[0])]
    #     prob_indices = valid_prob_indices[rand_perm]
    #     prob_indices = tuple([prob_indices[:,i] for i in range(4)])
    #     assert labels_batch[prob_indices].sum() == 0
    #     labels_batch[prob_indices] = 1
    #     rand_values = torch.rand((min(num_probs, valid_prob_indices.shape[0]), num_classes)).to(device)
    #     labels_batch[prob_indices[:-1]] *= rand_values
    #     labels_batch[prob_indices[:-1]] /= labels_batch[prob_indices[:-1]].sum(dim=-1, keepdim=True)
    ######

    return X_batch_out, y_batch_out, refs_batch.long(), labels_batch

@torch.no_grad()
def get_y_batch(X_batch, y_batch, refs_batch, labels_batch, reference_model, num_probs=None):

    assert batch_size == 1

    labels_batch = labels_batch.float()
    valid_prob_indices = (labels_batch == 0).nonzero()

    if num_probs is None:
        # num_probs = torch.randint(0, max_num_probs + 1, (1,)).item()
        num_probs = 2
        num_probs = min(num_probs, valid_prob_indices.shape[0])
    else:
        if valid_prob_indices.shape[0] < num_probs:
            return None, None

    if num_probs == 0:
        labels_batch[labels_batch == -1] = 0
        return labels_batch, F.one_hot(y_batch, num_classes=num_classes).float()

    rand_perm = torch.randperm(valid_prob_indices.shape[0])[:min(num_probs, valid_prob_indices.shape[0])]
    prob_indices = valid_prob_indices[rand_perm]
    prob_indices = tuple([prob_indices[:,i] for i in range(4)])

    assert labels_batch[prob_indices].sum() == 0
    labels_batch[prob_indices] = 1

    rand_values = torch.rand((min(num_probs, valid_prob_indices.shape[0]), num_classes)).to(device)
    labels_batch[prob_indices[:-1]] *= rand_values
    labels_batch[prob_indices[:-1]] /= labels_batch[prob_indices[:-1]].sum(dim=-1, keepdim=True)

    labels_batch[labels_batch == -1] = 0

    prob_indices_iterated = [tuple(prob_indices[i][j].item() for i in range(3)) for j in range(num_probs)]
    prob_indices_iterated = list(set(prob_indices_iterated))

    combinations = []
    probabilities = []

    def Recurse(comb, prob):

        n = len(comb)
        if n == len(prob_indices_iterated):
            combinations.append(comb)
            probabilities.append(prob)
            return
        
        row = labels_batch[prob_indices_iterated[n]]
        for i, el in enumerate(row):
            if el > 0:
                Recurse(comb + [i], prob * el)

    Recurse([], 1)

    assert 0.999 < sum(probabilities) < 1.001
    probabilities = torch.stack(probabilities).unsqueeze(-1)
    combinations = F.one_hot(torch.tensor(combinations), num_classes=num_classes).float().to(device)

    labels_total = labels_batch.repeat(len(combinations), 1, 1, 1)
    # labels_total[(slice(None),) + prob_indices[1:-1]] = combinations
    _,a,b = zip(*prob_indices_iterated)
    labels_total[:,a,b] = combinations

    model_mode = reference_model.training
    reference_model.eval()
    out = reference_model(X_batch.expand(len(combinations), -1), refs_batch.expand(len(combinations), -1, -1), labels_total)
    reference_model.train(model_mode)

    out = (probabilities * F.softmax(out, dim=-1)).sum(dim=0)

    return labels_batch, out

    # print(((labels_total == 0) | (labels_total == 1)).sum().item()/labels_total.numel())
    # print(prob_indices[0].shape[0])
    # print(labels_total[0].numel() - (labels_total[0] == labels_total[1]).sum().item())