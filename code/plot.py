import matplotlib.pyplot as plt
import os
import numpy as np

cur_dir = os.path.abspath( os.path.dirname(__file__) )
result_file = os.path.join(cur_dir, "cpu_results.txt")
res_list = ['shared_mem']

m_size = []
cpu_time = []
gpu_res = []
print(np.log(2.718))
tmp = []
with open(result_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        res = list(map(int ,line.split(' ')))
        m_size.append(res[0])      
        cpu_time.append(res[1])      
        tmp.append(res[2])
gpu_res.append(tmp)

for file_name in res_list: 
    shared_file = os.path.join(cur_dir, file_name+'.txt')
    tmp = []
    with open(shared_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res = list(map(int ,line.split(' ')))
            tmp.append(res[2])
    gpu_res.append(tmp)

gpu_res = np.array(gpu_res)
cpu_time = np.array(cpu_time)
m_size = np.array(m_size)
plt.plot(np.log2(m_size), np.log2(cpu_time))
for gpu_time in gpu_res:
    plt.plot(np.log2(m_size), np.log2(gpu_time))

plt.legend(['1 core cpu','global_mem']+res_list)
plt.show()