
from games.cartpole_wrapper import Cartpole_Wrapper
from games.pendulum_wrapper import Pendulum_Wrapper, Pendulum_Wrapper_discrete, Pendulum_Wrapper_no_norm
from games.lunarlander_wrapper import LunarLander_Wrapper, LunarLander_Wrapper_discrete
from games.frozenlake_wrapper import FrozenLake_Wrapper
from games.maze.maze_game import MazeGame
from ray.tune.registry import register_env


wrapper_switch = {
                  'CartPole-v1': Cartpole_Wrapper,
                  'Pendulum-v1_discrete': Pendulum_Wrapper_discrete,
                  'Pendulum-v1': Pendulum_Wrapper,
                  'Pendulum-v1-no-norm': Pendulum_Wrapper_no_norm,
                  'LunarLander-v2': LunarLander_Wrapper,
                  'LunarLander-v2_discrete': LunarLander_Wrapper_discrete,
                  'FrozenLake-v1': FrozenLake_Wrapper,
                  'Maze': MazeGame}

def create_env(config):
    
    try: 
        env = wrapper_switch[config.env]
    except:
        raise NotImplementedError(\
            f"Cannot create '{config.env}' - enviroment.")
    
    register_env(config.env, env)
    print("Training on {} - game.".format(config.env))
    