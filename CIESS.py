import numpy as np
import torch
from ActorCritic import Actor, CriticTD
from dataset_util.ImplicitCF import ImplicitCF
import model.Configurations as config
from Buffer import Buffer, normalise_actions
from RecSysEnv import RecSysEnv
from Graph import Graph
import matplotlib.pyplot as plt
from OUActionNoise import OUActionNoise


def policy(state, actor_model):
    state = torch.tensor(state, dtype=torch.float32).to(config.device)
    # sampling actions from μ
    pred = actor_model(state).squeeze()
    return pred.cpu().detach().numpy()


# select the indices for actions with max critic values
def pick_max_indices(actions, values):
    new_shape = (int(values.shape[0] / (config.RANDOM_WALK_STEPS + 1)), config.RANDOM_WALK_STEPS + 1)
    values_reshaped = values.reshape(new_shape)
    max_index = np.argmax(values_reshaped, axis=1)
    max_index = max_index + values_reshaped.shape[1] * np.arange(len(max_index))
    return actions[max_index]


# compute the critic values and return the actions with max critic values
def pick_max_actions(states, candidate_actions, critic_model):
    assert states.shape[1] == config.STATE_SIZE
    assert candidate_actions.shape[1] == config.RANDOM_WALK_STEPS + 1
    state_input = np.repeat(states, config.RANDOM_WALK_STEPS + 1, axis=0)
    state_input = torch.tensor(state_input, dtype=torch.float32).to(config.device)

    candidate_actions = candidate_actions.flatten()
    normalised_actions = np.expand_dims(normalise_actions(candidate_actions), 1)
    normalised_actions = torch.tensor(normalised_actions, dtype=torch.float32).to(config.device)

    # compute the critic values for each candidate actions
    critic_values_1, _ = critic_model([state_input, normalised_actions])

    return pick_max_indices(candidate_actions, critic_values_1.cpu().detach().numpy())


def select_actions(
        prev_state, action_graph,
        actor_model, critic_model,
        mode, noise=0, verbose=1
):
    # the actions here are noised
    proto_actions = policy(prev_state, actor_model)
    info = 'max candidate action = {:.2f}, min candidate action = {:.2f}'
    print(info.format(np.max(proto_actions), np.min(proto_actions)))

    # candidate actions: (batch size * steps)
    candidate_actions = action_graph.walk(proto_actions, config.RANDOM_WALK_STEPS)
    actions = pick_max_actions(prev_state, candidate_actions, critic_model)

    noised_actions = actions + noise
    if verbose == 1:
        print('{} actions {:.4f} {} noise {:.4f} = {:.4f}, max = {:.2f}, min = {:.2f}'.format(
            mode,
            np.mean(actions),
            '+' if noise > 0 else '-',
            abs(noise),
            np.mean(noised_actions),
            max(noised_actions),
            min(noised_actions)
        ))
    return np.clip(np.round(noised_actions).astype(np.int32), config.MIN_EMB_SIZE, config.MAX_EMB_SIZE)


def add_data(data, rl_history):
    count = 0
    for key in rl_history:
        rl_history[key].append(data[count])
        count += 1


def visualise_data(rl_history):
    for key in ['reward', 'action', 'quality']:
        plt.plot(rl_history[key + '_u'], label='user ' + key)
        plt.plot(rl_history[key + '_i'], label='item ' + key)
        if key == 'action':
            plt.plot(rl_history['sparsity'], label='sparsity')
        if key == 'reward':
            plt.plot(rl_history['avg_reward'], label='avg_reward')
        plt.legend()
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.show()
        plt.show()


def pretrain_recsys(env, dataset):
    pth = 'tmp/trained_{}_{}.pth'.format(env.base_model, dataset.dataset_type)
    best = 0
    patience = config.MAX_PATIENCE
    try:
        print('Loading the model')
        env.agent.load_state_dict(torch.load(pth))
    except:
        print('Failed to load the pretrained model. Now pretraining the model')
        while True:
            sampled_users = np.random.choice(
                dataset.user_vocab,
                round(config.SAMPLING_RATIO_USER * dataset.n_users),
                replace=False
            )
            sampled_items = np.random.choice(
                dataset.item_vocab,
                round(config.SAMPLING_RATIO_ITEM * dataset.n_items),
                replace=False
            )
            rq, _ = env.train_n_batches(3000, sampled_users, sampled_items, verbose=1)
            rq = sum(rq) / len(rq)
            if rq > best:
                best = rq
                patience = config.MAX_PATIENCE
                torch.save(env.agent.state_dict(), pth)
            else:
                patience -= 1
                if patience == 0:
                    break
        env.agent.load_state_dict(torch.load(pth))
        env.optimizer.param_groups[0]['lr'] = env.max_lr

    sampled_users = np.random.choice(dataset.user_vocab,
                                     round(config.SAMPLING_RATIO_USER * dataset.n_users), replace=False)
    all_items = np.array(dataset.item_vocab)
    user_metrics, item_metrics = env.eval_rec(sampled_users, all_items)
    env.user_peak_qualities[sampled_users] = np.maximum(user_metrics, 1e-5)
    env.item_peak_qualities[all_items] = np.maximum(item_metrics, 1e-5)


def train_rl(noise, ep, env,
             buffer_u, buffer_i,
             action_graph,
             actor_model_u,
             actor_model_i,
             critic_model_u, critic_model_i
             ):
    prev_state_u = env.get_state('user')
    prev_state_i = env.get_state('item')

    # actions for the sampled users
    actions_u = select_actions(prev_state_u, action_graph,
                               actor_model_u, critic_model_u,
                               'user', noise=noise
                               )
    actions_i = select_actions(prev_state_i, action_graph,
                               actor_model_i, critic_model_i,
                               'item', noise=noise
                               )
    mean_action_u = np.mean(actions_u)
    mean_action_i = np.mean(actions_i)
    sparsity = (sum(actions_u) + sum(actions_i)) / (config.MAX_EMB_SIZE * (env.dataset.n_users + env.dataset.n_items))

    # sampling users and items
    # sampling users
    if env.dataset.dataset_type == 'yelp':
        sampled_portion = round(config.SAMPLING_RATIO_USER * env.dataset.n_users)
        sampled_users = np.random.choice(env.dataset.user_vocab, sampled_portion, replace=False)
        sampled_portion = round(config.SAMPLING_RATIO_ITEM * env.dataset.n_items)
        sampled_items = np.random.choice(env.dataset.item_vocab, sampled_portion, replace=False)
    else:
        sampled_items = np.array(env.dataset.item_vocab)
        sampled_users = np.array(env.dataset.user_vocab)

    # execute the policy

    if env.base_model == 'ncf':
        state_u, state_i, reward_u, reward_i, quality_u, quality_i, sparsity = env.step(
            actions_u, actions_i,
            sampled_users, sampled_items
        )
        sparsity = sparsity[0]
    else:
        state_u, state_i, reward_u, reward_i, quality_u, quality_i = env.step(
            actions_u, actions_i,
            sampled_users, sampled_items
        )

    print('Current sparsity = {:.4f}'.format(sparsity))

    # store the transition in the buffer
    buffer_u.record((prev_state_u[sampled_users], actions_u[sampled_users], reward_u, state_u[sampled_users]))
    buffer_i.record((prev_state_i[sampled_items], actions_i[sampled_items], reward_i, state_i[sampled_items]))

    # update the policy
    buffer_u.learn('user', ep)
    buffer_i.learn('item', ep)

    return quality_u, quality_i, np.mean(reward_u), np.mean(reward_i), mean_action_u, mean_action_i, sparsity


def initialize_dataset(base_model, dataset_type):
    # setting up the dataset
    dataset = ImplicitCF(base_model, dataset_type)
    print('training set size = {0}, test set size = {1}, batch size = {2}'.format(
        dataset.train_size,
        dataset.test_size,
        config.BATCH_SIZE)
    )
    print('user size: {}, item size: {}'.format(
        dataset.n_users,
        dataset.n_items
    ))
    return dataset


def train_ddpg(_lambda, base_model, dataset, noise_type, visualisation=True):
    QUALITY_LOSS = True
    dataset_type = dataset.dataset_type

    # define the environment
    env = RecSysEnv(dataset, _lambda, base_model)

    # pretrain models
    pretrain_recsys(env, dataset)

    # networks for the user
    actor_model_u = Actor().to(config.device)
    critic_model_u = CriticTD().to(config.device)
    target_actor_u = Actor().to(config.device)
    target_critic_u = CriticTD().to(config.device)

    # networks for the item
    actor_model_i = Actor().to(config.device)
    critic_model_i = CriticTD().to(config.device)
    target_actor_i = Actor().to(config.device)
    target_critic_i = CriticTD().to(config.device)

    buffer_u = Buffer(
        target_actor=target_actor_u,
        target_critic=target_critic_u,
        actor_model=actor_model_u,
        critic_model=critic_model_u,
        mode='user', dataset=dataset
    )

    buffer_i = Buffer(
        target_actor=target_actor_i,
        target_critic=target_critic_i,
        actor_model=actor_model_i,
        critic_model=critic_model_i,
        mode='item', dataset=dataset
    )

    action_graph = Graph(config.MAX_EMB_SIZE)

    # data visualisation
    rl_history = {
        'sparsity': [], 'action_u': [], 'action_i': [],
        'quality_u': [], 'quality_i': [],
        'reward_u': [], 'reward_i': [], 'avg_reward': []
    }
    sparsity_goals = {0.05: [0] * 4, 0.1: [0] * 4, 0.2: [0] * 4}
    for ep in range(config.M):

        # define the environment
        env = RecSysEnv(
            dataset, _lambda, base_model,
            user_peak_quality=env.user_peak_qualities,
            item_peak_quality=env.item_peak_qualities
        )
        assert np.mean(env.agent.user_sizes) == np.mean(env.agent.item_sizes) == config.MAX_EMB_SIZE

        print('=' * 30 + 'episode ' + str(ep) + '=' * 30)
        print('##################### TRAIN #####################')
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=config.OU_STD_DEV)
        for loop in range(config.T):
            print('{}ep: {}, loop:{}{}'.format('-' * 10, ep, loop, '-' * 10))
            if noise_type == 'normal':
                noise = np.random.normal(0, 6)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-11, 11)
            else:
                noise = ou_noise()[0]

            # train the user networks
            quality_u, quality_i, \
                mean_reward_u, mean_reward_i, \
                mean_actions_u, mean_actions_i, \
                sparsity = train_rl(
                    noise, ep, env,
                    buffer_u, buffer_i,
                    action_graph,
                    actor_model_u, actor_model_i,
                    critic_model_u, critic_model_i
                )

            # store the best results
            quality_u_mean = np.mean(quality_u)
            alias_temp = 'tmp/{}_sizes_lambda{}_ep{}_loop{}_basemodel{}_noise{}_dataset{}.npy'
            alias_user = alias_temp.format('user', _lambda, ep, loop, base_model, noise_type, dataset_type)
            np.save(alias_user, env.agent.user_sizes)

            alias_item = alias_temp.format('item', _lambda, ep, loop, base_model, noise_type,
                                           QUALITY_LOSS, dataset_type)
            np.save(alias_item, env.agent.item_sizes)
            for goal in sparsity_goals:
                if sparsity < goal and sparsity_goals[goal][2] < quality_u_mean:
                    sparsity_goals[goal] = [ep, loop, quality_u_mean, sparsity]
                    break
            print('sparsity goals: ', sparsity_goals)

            if loop == config.T - 1:
                add_data([
                    sparsity * 100,
                    mean_actions_u, mean_actions_i,
                    quality_u_mean, np.mean(quality_i),
                    mean_reward_u, mean_reward_i,
                    (mean_reward_u + mean_reward_i) / 2],
                    rl_history
                )

        # data visualisation
        if ep == config.M - 1 and visualisation:
            visualise_data(rl_history)

    return sparsity_goals


