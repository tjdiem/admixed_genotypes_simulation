from globals import *

OHE_values = torch.arange(3**4).reshape(3, 3, 3, 3) # SOI value, ref value, class we are in, labeled class of ref
OHE_values[:, :, 2] = torch.flip(OHE_values[:, :, 0], dims=(2,))
OHE_values[2] = torch.flip(OHE_values[0], dims=(0,))
unique_elements, inverse_indices = torch.unique(OHE_values, return_inverse=True)
OHE_values = torch.arange(len(unique_elements))[inverse_indices]
OHE_values = OHE_values.flatten()

assert 36 == OHE_values.max() + 1

Transition = torch.zeros((num_classes**4, n_embd))
Transition[torch.arange(num_classes**4).long(), OHE_values] = 1 #dype = long

#####
def translate(SOI, ref, clas, label):
    val = 27 * SOI + 9 * ref + 3 * clas + 1 * label
    val = torch.tensor(val)
    val = F.one_hot(val, num_classes=num_classes ** 4)
    val = val.float() @ Transition
    print(torch.argmax(val))

translate(1, 2, 0, 1)
translate(1, 0, 2, 1)