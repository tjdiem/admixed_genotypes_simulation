import sys

base_pairs = 50000000
morgans = 1
base_calling_error_rate = 0.0001
min_freq_difference = 0.2
max_freq_difference = 0.8
morgan_cutoff = 0.00002

parental_samples = 50




with open(sys.argv[1]) as macs_output:
    macs_output = macs_output.readlines()

    site_pos = []
    samples = []

    for line in macs_output:
       
        line = line.split()
        if line[0] != "SITE:":
            continue
        
        site_pos.append(float(line[2]))
        samples.append(line[4])





num_samples = len(samples[0])//2

parental_1_count = []
parental_2_count = []



for i in range(len(site_pos)):
    parental_1_count.append(samples[i][0:parental_samples].count("0"))
    parental_2_count.append(samples[i][num_samples:num_samples+parental_samples].count("0"))





#Site filtering
good = []


#non informative positions
for i in range(len(site_pos)):
    good.append(True)
    if max_freq_difference < abs(parental_1_count[i]/50 - parental_2_count[i]/50) or abs(parental_1_count[i]/50 - parental_2_count[i]/50) < min_freq_difference:
        good[i] = False



#LD pruning
last_morgan = -1

for i in range(len(site_pos)):
    if good[i]:
        if site_pos[i]*morgans - last_morgan < morgan_cutoff:
            good[i] = False
        else:
            last_morgan = site_pos[i]*morgans



last_morgan = 0

print("")

for i in range(len(site_pos)):
    if good[i]:
        print("1", int(site_pos[i]*base_pairs),parental_1_count[i], parental_samples - parental_1_count[i], parental_2_count[i], parental_samples - parental_2_count[i], site_pos[i]*morgans - last_morgan,
        samples[i][parental_samples:num_samples], samples[i][num_samples + parental_samples:])


