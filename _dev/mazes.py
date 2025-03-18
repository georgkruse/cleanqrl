import numpy as np


def optimal_policy_crawford(n):
    optimal_map = ((4,), (3,), (3,), (3, 1), (3,))
    for n in range(n - 2):
        optimal_map += ((1,), (3, 1), tuple(), (1,), (3, 1))
    optimal_map += ((1,), (3, 1), (3,), (1,), (3, 1))
    return optimal_map


def create_crawford(P, A, W, R, n):
    obs = np.zeros((n, 5))
    for n in range(n - 2):
        obs[n + 1, 2] = W
    obs[0, 0] = R
    obs[n - 1, 2] = P
    obs[n - 1, 4] = A
    return obs


def create_mueller_3x3(P, A, W, R, **kwargs):
    obs = np.zeros((3, 3))
    obs[0, 0] = R
    x, y = 2, 2
    # x, y = np.random.randint(0, 3, 2)
    if x == 0 and y == 0:
        x, y = 2, 2
    obs[y, x] = A
    return obs


def optimal_policy_mueller_3x3(**kwargs):
    optimal_map = (
        (4,),
        (3,),
        (3,),
        (1,),
        (3, 1),
        (3, 1),
        (1,),
        (3, 1),
        (3, 1),
    )
    return optimal_map


def create_neumann_a(P, A, W, R, **kwargs):
    obs = np.array([[0, A, 0, 0], [0, W, 0, P], [0, 0, 0, R]])
    return obs


def optimal_policy_neumann_a(**kwargs):
    optimal_map = (
        (2, 0),
        (2,),
        (0,),
        (3,),
        (0,),
        tuple(),
        (0,),
        (0,),
        (2,),
        (2,),
        (2,),
        (4,),
    )
    return optimal_map


def create_neumann_b(P, A, W, R, **kwargs):
    obs = np.array([[R, 0, 0, 0, 0], [0, 0, W, 0, 0], [0, 0, 0, 0, A]])
    return obs


def optimal_policy_neumann_b(**kwargs):
    optimal_map = ((4,), (3,), (3,), (3, 1), (3,))
    optimal_map += ((1,), (3, 1), tuple(), (1,), (3, 1))
    optimal_map += ((1,), (3, 1), (3,), (1,), (3, 1))
    return optimal_map


def create_neumann_c(P, A, W, R, **kwargs):
    obs = np.array([[0, A, 0, 0, P], [0, 0, P, 0, P], [0, W, W, R, 0], [0, 0, 0, 0, 0]])
    return obs


def optimal_policy_neumann_c(**kwargs):
    optimal_map = ((2, 0), (2,), (2,), (0,), (3,))
    optimal_map += ((2, 0), (1,), (2,), (0,), (3, 0))
    optimal_map += ((0,), tuple(), tuple(), (4,), (3,))
    optimal_map += ((2,), (2,), (2,), (1,), (3, 1))
    return optimal_map


def create_neumann_d(P, A, W, R, **kwargs):
    obs = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, R],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, W, 0, 0, 0, P, P, 0],
            [0, 0, W, A, W, 0, 0, P, P, 0],
            [0, 0, 0, 0, 0, 0, 0, P, P, 0],
            [0, W, W, W, W, W, 0, P, P, 0],
            [0, W, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, W, 0, 0, 0, 0, R, 0],
        ]
    )
    return obs
