import pickle
with open('records_2024-02-03_12-18-49_noiseratio_10_whole_matching.pkl','rb') as f:
    records = pickle.load(f)

steps = records['steps']
costs = records['costs']
times = records['times']
times_lower_bound = records['times_lower_bound']
times_infer = records['times_infer']

def find_indices(lst):
    return [index for index, value in enumerate(lst) if value != -1]
indices = find_indices(steps)


print(f'Number of cases sloved within 5000 steps: {str(len(indices))}')
print(f'Ratio of cases sloved within 5000 steps: {len(indices)/len(steps)}')

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

