import numpy as np
import random
import config
import pandas as pd
import sys
from artifacts import Goal
from matplotlib import pyplot
import seaborn as sns
from environment import Environment, Quit, Action

environment = Environment(f'maps/map.txt')
dict_for_moves = {
    Action.UP: "Up",
    Action.LEFT: "Left",
    Action.DOWN: "Down",
    Action.RIGHT: "Right"
}

duzina = len(environment.field_map)
sirina = len(environment.field_map[0])


def train():
    df_stats = pd.DataFrame({'episode': [],
                             'return_reward': [],
                             'steps': []})
    data = {'Up': [],
            'Left': [],
            'Down': [],
            'Right': []}
    q_tab = pd.DataFrame(data)
    rewards = np.empty((duzina, sirina))

    # popunjavanje rewards matrice
    for i in range(0, duzina):
        for j in range(0, sirina):
            q_tab.loc[i * sirina + j] = [0., 0., 0., 0.]
            if [i, j] == environment.artifacts_map[Goal.kind()].get_position():
                rewards[i][j] = 100
            else:
                rewards[i][j] = environment.field_map[i][j].reward()

    # praljenje q tabele
    num_episodes = 3000
    max_steps = 100
    learning_rate = 0.1
    gamma = 0.95
    epsilon_max = 1.0
    epsilon_min = 0.005
    epsilon_decay_rate = 0.001
    for episode in range(num_episodes):
        environment.reset()
        eps = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_decay_rate * episode)
        old_pos = environment.get_agent_position().copy()
        ep_step = 0
        ep_rew = 0
        for step in range(max_steps):
            prob = random.uniform(0, 0.9)
            ep_step += 1
            arr = q_tab.loc[old_pos[0] * sirina + old_pos[1]]
            new_maw = np.argmax(arr)
            next_act = Action(new_maw) if prob > eps else environment.get_random_action()
            pos, reward, end_pos = environment.step(next_act)

            if old_pos == pos:
                reward = -100
            else:
                reward = rewards[pos[0], pos[1]]
            ep_rew += reward
            old_q = q_tab.loc[old_pos[0] * sirina + old_pos[1], dict_for_moves[next_act]]
            temp_diff = reward + (gamma * np.max(q_tab.loc[pos[0] * sirina + pos[1]])) - old_q

            new_q = old_q + learning_rate * temp_diff
            q_tab.loc[old_pos[0] * sirina + old_pos[1], dict_for_moves[next_act]] = new_q
            if end_pos:
                break
            old_pos[0] = pos[0]
            old_pos[1] = pos[1]
        df_stats.loc[episode] = [episode, ep_rew, ep_step]
    print("Training complete!")
    print("==========================")
    print("Q Table:")
    print(q_tab)
    print("==========================")
    q_tab.to_csv('q_tab.csv')
    df_stats.to_csv('stats.csv')


def evaluate():
    pass


def results():
    evaluate()
    df_stats = pd.read_csv('stats.csv')
    df_stats = df_stats.groupby(np.arange(len(df_stats.index)) // 40).mean()
    pyplot.figure("Average reward per episode")
    plot_1 = sns.lineplot(data=df_stats, x='episode', y='return_reward', marker='o',
                          markersize=5, markerfacecolor='red')
    plot_1.get_figure().savefig('reward_graph.png')
    pyplot.show()
    pyplot.figure("Average steps per episode")
    plot_1 = sns.lineplot(data=df_stats, x='episode', y='steps', marker='o',
                          markersize=5, markerfacecolor='red')
    plot_1.get_figure().savefig('steps_graph.png')
    pyplot.show()
    pass


def simulate():
    try:
        environment.reset()
        position = environment.get_agent_position().copy()
        environment.render(config.FPS)
        q_table = pd.read_csv("q_tab.csv")
        while True:
            arr = q_table.loc[position[0] * sirina + position[1]][1:5]
            action = np.argmax(arr)
            position, _, done = environment.step(Action(action))
            environment.render(config.FPS)
            if done:
                break
    except Quit:
        pass


if len(sys.argv) == 1:
    # If no arguments provided, call functions in the specified order
    train()
    results()
    simulate()
elif len(sys.argv) == 2:
    # If one argument provided, check its value and call the corresponding function
    if sys.argv[1] == "simul":
        simulate()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "result":
        results()
    else:
        print("Invalid argument. Please use 'simul', 'train', 'result', or provide no arguments.")
else:
    print("Invalid number of arguments. Please use 'simul', 'train', 'result', or provide no arguments.")
