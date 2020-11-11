"""Run script for LOLA-DiCE on IPD."""

import click
from datetime import datetime

import tensorflow as tf

from lola_dice.envs import *
from lola_dice.policy import SimplePolicy, MLPPolicy, RecurrentPolicy, ConvPolicy
from lola_dice.rpg import train


@click.command()
@click.option("--use-dice/--no-dice", default=True,
              help="Whether to use the DiCE operator in the policy objective.")
@click.option("--use-opp-modeling/--no-opp-modeling", default=False,
              help="Whether to use opponent modeling.")
@click.option("--batch-size", default=64)
@click.option("--epochs", default=200)
@click.option("--runs", default=5)
# @click.option("--save-dir", default="results_ipd")

@click.option("--exp_name", type=str, default="IPD",
              help="Name of the experiment (and correspondingly environment).")
@click.option("--trace_length", type=int, default=None,
              help="Lenght of the traces.")
@click.option("--gamma", type=float, default=None,
              help="Discount factor.")
@click.option("--grid_size", type=int, default=3,
              help="Grid size of the coin game (used only for coin game).")

def main(use_dice, use_opp_modeling, epochs, batch_size, runs, exp_name, trace_length, gamma, grid_size):

    def make_simple_policy(ob_size, num_actions, prev=None, root=None):
        return SimplePolicy(ob_size, num_actions, prev=prev)

    def make_mlp_policy(ob_size, num_actions, prev=None):
        return MLPPolicy(ob_size, num_actions, hidden_sizes=[64], prev=prev)

    def make_conv_policy(ob_size, num_actions, prev=None):
        return ConvPolicy(ob_size, num_actions, hidden_sizes=[16,32], prev=prev)

    def make_adam_optimizer(*, lr):
        return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                name='Adam')

    def make_sgd_optimizer(*, lr):
        return tf.train.GradientDescentOptimizer(learning_rate=lr)

    n_agents = 2
    # env = IPD(max_steps=150, batch_size=batch_size)

    if exp_name in {"IPD", "IMP"}:
        # num_episodes = 600000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        # batch_size = 4000 if batch_size is None else batch_size
        base_lr = 1.0
        make_optim = make_sgd_optimizer
        save_dir = "dice_results_ipd"

    elif exp_name == "CoinGame":
        # num_episodes = 100000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        # batch_size = 4000 if batch_size is None else batch_size
        base_lr = 0.005
        epochs *= 10
        make_optim = make_adam_optimizer
        save_dir = "dice_results_coin_game"

    # Instantiate the environment
    if exp_name == "IPD":
        env = IPD(max_steps=trace_length, batch_size=batch_size)
        gamma = 0.96 if gamma is None else gamma
        policy_maker = make_simple_policy
    elif exp_name == "IMP":
        env = IMP(trace_length)
        gamma = 0.9 if gamma is None else gamma
        policy_maker = make_simple_policy
    elif exp_name == "CoinGame":
        env = CG(trace_length, batch_size, grid_size)
        gamma = 0.96 if gamma is None else gamma
        policy_maker = make_conv_policy
        env.seed(SEED)
    # elif exp_name == "AssymCoinGame":
    #     env = AssymCG(trace_length, batch_size, grid_size)
    #     gamma = 0.96 if gamma is None else gamma

    else:
        raise ValueError(f"exp_name: {exp_name}")

    start_time_str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    for r in range(runs):
        print("-" * 10, "Run: %d/%d" % (r + 1, runs), "-" * 10)
        train(env, policy_maker, #make_conv_policy, #make_simple_policy,
              make_optim, #make_sgd_optimizer,
              epochs=epochs,
              gamma=gamma, #.96,
              lr_inner=base_lr*.1,
              lr_outer=base_lr*.2,
              lr_value=base_lr*.1,
              lr_om=base_lr*.1,
              inner_asymm=True,
              n_agents=n_agents,
              n_inner_steps=2,
              value_batch_size=16,
              value_epochs=0,
              om_batch_size=16,
              om_epochs=0,
              use_baseline=False,
              use_dice=use_dice,
              use_opp_modeling=use_opp_modeling,
              save_dir=f'{save_dir}/{start_time_str}/run-{r + 1}')


SEED = 2020
if __name__ == '__main__':
    main()
