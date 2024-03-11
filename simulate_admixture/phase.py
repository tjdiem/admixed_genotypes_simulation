import sys

split_file = sys.argv[1]
phase_file = sys.argv[2]

split_file = open(split_file).readlines()


if len(split_file[-1]) < 2:
    del split_file[-1]



# individuals are two rows each
# first is split, 
# second is phase, the phase that starts at the split point above it


def check(line, x):
    if x < len(line):
        return float(line[x])
    else:
        return 100000

phase_file = open(phase_file, "w")

for i in range(len(split_file) // 2):
    line_1 = split_file[i*2].split()
    line_2 = split_file[i*2 + 1].split()

    split_line = []
    phase_line = []

    morg = 0

    anc_0 = 0
    anc_1 = 0

    j = 0
    k = 0

    while True:
        if check(line_1,j) < check(line_2,k):
            anc_0 = 1 - anc_0

            split_line.append(line_1[j])
            phase_line.append(str(anc_0 + anc_1))

            j += 1
        elif check(line_1,j) > check(line_2,k):
            anc_1 = 1 - anc_1

            split_line.append(line_2[k])
            phase_line.append(str(anc_0 + anc_1))

            k += 1
        else:
            if len(line_1) <= j and len(line_2) <= k:
                break

            anc_0 = 1 - anc_0
            anc_1 = 1 - anc_1

            split_line.append(line_1[j])
            phase_line.append(str(anc_0 + anc_1))

            j += 1
            k += 1

            
    
    phase_file.write("\t".join(split_line)+"\n")
    phase_file.write("\t".join(phase_line)+"\n")


