import sys
from math import *


prop = float(sys.argv[1])
time = float(sys.argv[2])
sampled_individuals = int(sys.argv[3])
id = sys.argv[4]

# generate selection file, currently empty, no selection
selection = open("SELAM_inputs/selection_"+id, "w")
selection.close()


# generate output file (this determines time and number of samples)
output = open("SELAM_inputs/output_"+id, "w")
output.write(str(floor(time))+"\t0\t"+str(sampled_individuals//2)+"\t"+str(ceil(sampled_individuals/2))+"\tselam_output_"+id+"\n")
output.close()


#generate demography file
demo = open("SELAM_inputs/demography_"+id, "w")
demo.write("pop1\tpop2\tsex\t0\t1\n")
demo.write("0\t0\tA\t10000\t10000\n")
demo.write("0\ta0\tA\t"+str(prop)+"\t0\n")
demo.write("0\ta1\tA\t"+str(1-prop)+"\t0\n")
demo.close()
