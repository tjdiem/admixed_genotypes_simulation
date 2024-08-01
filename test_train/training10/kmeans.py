from globals import *
from processing import *
from sklearn.cluster import KMeans
from itertools import permutations

pos_ind = input_size // 2
window = 500
threshold = 400

classes = [0,1,2]

all_perms = list(set(permutations(classes)))
print(all_perms)


panel_data = [convert_panel(panel_dir + "panel_" + str(i)) for i in range(num_files)]
X = [x for x, _ in panel_data]
positions = [pos for _, pos in panel_data]

y = [convert_split(split_dir + "split_" + str(i), positions[i]) for i in range(num_files)]

X = torch.tensor(X) # num_files, n_ind_adm, input_size
y = torch.tensor(y) # num_files, 2 * n_ind_adm, input_size
y = y[:,::2] + y[:,1::2] # unphase ancestry labels # same shape as X

window_slice = slice(input_size // 2 - window // 2, input_size // 2 + window // 2 + 1)

X = X[:, window_slice, :-1]
y = y[:, window_slice]

Scores = []
for i in range(X.shape[0]):
    y_example = []
    valid_indices = []
    for j in range(X.shape[1]):
        for k in range(num_classes):
            if (y[i,j] == k).sum().item() >= threshold:
                valid_indices.append(j)
                y_example.append(k)

    
    x_example = X[i,valid_indices]
    print(x_example.shape)
    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    kmeans.fit(x_example)
    assignments = kmeans.labels_
    scores = []

    for j in range(len(all_perms)):
        assignments_perm = [all_perms[j][a] for a in assignments]
        scores.append(sum([a == b for a, b in zip(assignments_perm, y_example)]))

    Scores.append(max(scores)/len(assignments))

print(sum(Scores) / len(Scores))


print(X.shape)
print(y.shape)


