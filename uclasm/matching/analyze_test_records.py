import pickle
import numpy as np
with open('./results/records_MSRC_21_noiseratio_5_RL_iteration.pkl','rb') as f:
    records = pickle.load(f)

steps = records['steps']
costs = records['costs']
costs_wo_bt = records['costs_wo_bt']
times = records['times']
times_lower_bound = records['times_lower_bound']
times_infer = records['times_infer']

def find_indices(lst):
    return [index for index, value in enumerate(lst) if value != -1]
indices = find_indices(steps)


print(f'Number of cases sloved within 600 seconds: {str(len(indices))}')
print(f'Ratio of cases sloved within 600 seconds: {len(indices)/len(steps)}')

avg_cost = sum(costs)/len(costs)
print(f'Average GED: {str(avg_cost)}')
std_dev = np.std(costs)
print("std:", std_dev)

avg_cost = sum(costs_wo_bt)/len(costs_wo_bt)
print(f'Average costs_wo_bt: {str(avg_cost)}')

steps_solved= [steps[i] for i in indices]
avg_step = sum(steps_solved)/len(indices)
print(f'Average steps in sloved cases: {str(avg_step)}')





times_solved= [times[i] for i in indices]
avg_time = sum(times_solved)/len(indices)
print(f'Average time in sloved cases: {str(avg_time)}')

ave_time_lower_bound = sum(times_lower_bound)/len(times_lower_bound)
print(f'Average time to get lower bound: {str(ave_time_lower_bound)}')

ave_time_infer = sum(times_infer)/len(times_infer)
print(f'Average time to get infer: {str(ave_time_infer)}')

times_get_lower_bound = len(times_lower_bound)/len(steps)
print(f'How many times to calculate lowerbound each graph: {str(times_get_lower_bound)}')

times_infer = len(times_infer)/len(steps)
print(f'How many times to infer each graph: {str(times_infer)}')

