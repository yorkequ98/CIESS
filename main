from CIESS import train_ddpg, retrain, initialize_dataset
import Configurations as config

model = 'ngcf'
noise = 'normal'
_lambda = 0.01

# run the following to train the RL policy and produce embedding sizes
assert config.STATE_SIZE == 3
assert config.MIN_EMB_SIZE == 1
assert config.RANDOM_WALK
assert config.MAX_PATIENCE == 2
assert config.SAMPLING_RATIO_USER == 1.0
assert config.SAMPLING_RATIO_ITEM == 1.0
data = initialize_dataset(model, 'ml-1m')
goals = train_ddpg(_lambda=_lambda, base_model=model, visualisation=True, noise_type=noise, dataset=data)
data.switch_to_test_mode()


# run this to retrain the recommenders to get the final recommendation performance
def find_bests(_lambda, k=3):
    lines = []
    model = 'ngcf'
    data = initialize_dataset(model, 'ml-1m')
    data.switch_to_test_mode()
    with open('../data/ngcf/output_0.05.txt'.format(_lambda), 'r') as f:
        lines.extend([line[:-1] for line in f])
    sparsity = -1
    avg = -1
    goals = {0.05: [], 0.1: [] , 0.2: []}
    ep = -1
    loop = -1
    for line in lines:
        if 'ep' in line and 'loop' in line:
            ep = int(line.split(',')[0].split(':')[1])
            loop = int(line.split(',')[1].split(':')[1][:-10])
            sparsity = -1
            avg = -1

        if 'AVG' in line:
            avg = float(line.split(' ')[-1])
        if 'Current sparsity' in line:
            sparsity = float(line.split('=')[-1])
        if avg != -1 and sparsity != -1 and 'sparsity goals' in line:
            for level in goals:
                if sparsity < level:
                    goals[level].append([ep, loop, avg, sparsity])
                    break
    for goal in goals:
        candidate = goals[goal]
        sorted_cand = sorted(
            candidate,
            key=lambda tup: tup[2],
            reverse=True
        )
        goals[goal] = sorted_cand

        for tup in sorted_cand[:k]:
            print(tup)
            if goal == 0.2:
                retrain(ep=tup[0], loop=tup[1], _lambda=_lambda, base_model=model, noise='normal', dataset=data)

    return goals

find_bests(0.05)
