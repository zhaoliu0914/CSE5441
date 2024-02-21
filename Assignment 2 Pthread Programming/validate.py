import sys
import re

def readList(path):
    f = open(path, 'r')
    ans = []
    for line in f.readlines():
        ans.append(int(line))
    f.close()
    return ans

def parseOutput(path):
    f = open(path, 'r')
    num_consumer = 0
    time = 0
    consume_numbers = []
    produce_numbers = []
    for line in f.readlines():
        try :
            line = line.strip()
            if line.startswith('main:'):
                num_consumer = int(line.split()[-1])
            elif 'inserting' in line:
                produce_numbers.append(int(line.split()[-4]))
            elif 'extract' in line:
                consume_numbers.append(int(line.split()[-3]))
            elif 'real' in line:
                time_split = re.split('[ms]', line.split()[1])
                time = float(time_split[0]) * 60
                time += float(time_split[1])
        except Exception as e:
            print("Experienced exception when parsing {}. Please check the output for abnormal output.".format(path))
            print(e)
    f.close()
    cons_ans = []
    prod_ans = []
    exit_nums = 0

    for n in produce_numbers:
        if n != -1:
            prod_ans.append(n)

    for n in consume_numbers:
        if n == -1:
            exit_nums += 1
        else:
            cons_ans.append(n)

    return num_consumer, prod_ans, cons_ans, exit_nums, time

path = sys.argv[1]
program = sys.argv[2]
data_list = sys.argv[3]
num_consumer = int(sys.argv[4])

input_data = readList(path + "/" + data_list)
fname = "{}-{}-output-{}.txt".format(program, data_list, num_consumer)
# print(fname)
cons_num, prod_list, cons_list, exit_nums, time = parseOutput(path + "/output/" + fname)
succ_str = "Success"
if cons_num != num_consumer:
    print("Expect {} consumers, but get {}".format(num_consumer, cons_num))
    succ_str = "Failure"
if sorted(prod_list) != sorted(input_data):
    print("Input and produced list have different elements")
    print(sorted(prod_list))
    print(sorted(input_data))
    succ_str = "Failure"
if sorted(cons_list) != sorted(input_data):
    print("Input and consumed list have different elements")
    print(sorted(cons_list))
    print(sorted(input_data))
    succ_str = "Failure"
if exit_nums != num_consumer:
    print("Consumer {}, but {} exit".format(num_consumer, exit_nums))
    succ_str = "Failure"

print("{}: {}.c ran with {} consumers for {} in {} seconds".format(succ_str, program, num_consumer, data_list, time))
