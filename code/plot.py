import matplotlib.pyplot as plt
import os
import numpy as np

cur_dir = os.path.abspath( os.path.dirname(__file__) )
result_file = os.path.join(cur_dir, "cpu_results.txt")
res_list = ['floyd_gpu','shared_mem', 'block_size_32', 'recursive_unroll']
#res_list = []

# global_mem, res_list, cpu_time
mask = [0,0,0,0,1,1]

m_size = []
cpu_time = []
gpu_res = []
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


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,}

gpu_res.append(cpu_time)
gpu_res = np.array(gpu_res)
#gpu_res = np.array(gpu_res)
#cpu_time = np.array(cpu_time)
m_size = np.array(m_size)
#plt.plot(np.log10(m_size), np.log10(cpu_time))
for index, gpu_time in enumerate(gpu_res):
    if mask[index]==1:
        plt.plot(np.log10(m_size), np.log10(gpu_time))
        print(gpu_time[-1])

legend = ['global_mem']+res_list+['1 core cpu']
mask = np.array(mask)
index = np.where(mask==1)[0]
plt.legend([legend[le] for le in index])
plt.xlabel('number of vertices (log space)', font1)
plt.ylabel('running time in millisecond (log space)', font1)
plt.show()

