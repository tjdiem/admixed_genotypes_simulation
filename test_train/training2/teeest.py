amounts = [0, 0, 0]

for i in range(200):
    with open("../../simulate_admixture/phases/phase_" + str(i), "r") as f:
        lines = f.readlines()[1::2]

    for l in lines:
        line = l.split("\t") 
        for j in range(3):
            amounts[j] += line.count(str(j))

m = max(amounts)
for i in range(3):
    amounts[i] /= m
print(amounts)