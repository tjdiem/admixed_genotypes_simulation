import torch
import time

which = 3

if which == 0:
    # save tensor
    a = torch.randint(0, 3, (400, 1000)).to(torch.int8)
    start_time = time.time()
    torch.save(a, "a.pt")

if which == 3:
    # save stacked tensor
    a = torch.randint(0, 3, (400, 1000)).to(torch.uint8)
    start_time = time.time()
    aclone = a.clone()
    a = a[...,::5] + 3 * a[...,1::5] + 9 * a[...,2::5] + 27 * a[...,3::5] + 81 * a[...,4::5]
    torch.save(a, "a_stacked.pt")

if which == 2:
    # load tensor
    start_time = time.time()
    b = torch.load("a.pt")

if which == 3:
    # load stacked tensor
    start_time = time.time()
    b = torch.load("a_stacked.pt")
    c = torch.zeros(*b.shape[:-1], 5 * b.shape[-1], dtype=torch.int8)
    for i in range(4):
        c[...,i::5] = b % 3
        b //= 3
    c[...,4::5] = b
    

# print(a[0])
# print(aclone[0])
# print(c[0])
# print(aclone[0] == c[0])

print(time.time() - start_time)