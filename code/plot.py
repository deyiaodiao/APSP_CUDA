import matplotlib.pyplot as plt
import os
import numpy as np

cur_dir = os.path.abspath( os.path.dirname(__file__) )
result_file = os.path.join(cur_dir, "cpu_results.txt")
shared_file = os.path.join(cur_dir, "shared_results.txt")

m_size = []
cpu_time = []
gpu_global = []
gpu_shared = []

with open(result_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        res = list(map(int ,line.split(' ')))
        m_size.append(res[0])      
        cpu_time.append(res[1])      
        gpu_global.append(res[2])

with open(shared_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        res = list(map(int ,line.split(' ')))
        gpu_shared.append(res[2])

gpu_global = np.array(gpu_global)
cpu_time = np.array(cpu_time)
m_size = np.array(m_size)
plt.plot(np.log(m_size), np.log(cpu_time))
plt.plot(np.log(m_size), np.log(gpu_global))
plt.plot(np.log(m_size), np.log(gpu_shared))
plt.legend(['1 core cpu','global gpu', 'shared gpu'])
plt.show()