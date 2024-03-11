import sys


panel_template = sys.argv[1]
selam_output = sys.argv[2]
diploids_sampled = int(sys.argv[3])



# list of lists of split points
# Ancestry starts at 0 (a split point at 0 is inserted if chromosme starts at 1)
# length is 2 * diploids_sampled
split_points = []



# Condense selam output into list of list of split points
with open(selam_output) as selam:
    lines = selam.readlines()

    chromosome = 0

    split_points.append([])

    for i in range(len(lines)):

        if lines[i][0] == '#':
            if lines[i-1][0] != "#":
                chromosome += 1
                if chromosome == 2 * diploids_sampled:
                    break
                split_points.append([])
            continue
        
        line = lines[i].split()

        if line[6] != '0' or line[7] != '0':
            split_points[chromosome].append(float(line[7]))





split_points_index = [0] * len(split_points)

with open(panel_template) as template:
    lines = template.readlines()

    last_morgan = 0

    for line in lines[1:]:
        line = line.split()

        morgans = float(line[6])

        #update split point index
        for i in range(len(split_points)):
            if split_points_index[i] < len(split_points[i]) and split_points[i][split_points_index[i]] < morgans:
                split_points_index[i] += 1
        

        sample_index = [0,0]
        
        #loop through diploids
        for i in range(len(split_points)//2):
            

            allele_1_count = 0

            chrom_ancestries = [ split_points_index[i*2] % 2 , split_points_index[i*2 + 1] % 2]


            allele_1_count += int(line[7 + chrom_ancestries[0]][sample_index[chrom_ancestries[0]]])
            sample_index[chrom_ancestries[0]] += 1

            allele_1_count += int(line[7 + chrom_ancestries[1]][sample_index[chrom_ancestries[1]]])
            sample_index[chrom_ancestries[1]] += 1

            line.append(str(2 - allele_1_count))
            line.append(str(allele_1_count))
        
        line[6] = str(morgans - last_morgan)
        last_morgan = morgans

        del line[7]
        del line[7]

        print("\t".join(line))


