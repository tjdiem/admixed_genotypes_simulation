from globals import * 
from processing import *

Data = [convert_output_file(panel_dir + "panel_" + str(i), phase_dir + "phase_" + str(i)) for i in range(num_files)]

X = [xx for xx, _ in Data if xx is not None]
y = [yy for _, yy in Data if yy is not None]

X = torch.tensor(X)
y = torch.tensor(y)

i = 4
print((X[i,:,25] - X[i,:,34]).int())
print(y[i,:,25] - y[i,:,34])

# print(X[i,:,34])
# print(y[i,:,34])