from ray.tune import choice, uniform, grid_search, loguniform

switch = {'choice': choice, 'uniform': uniform, 'grid_search': grid_search, 'loguniform': loguniform}

def add_hyperparameters(conf):
    for key, _ in conf.items():
        if isinstance(conf[key], list):
            if len(conf[key]) > 0 and not isinstance(conf[key][0], list):
                if conf[key][0] in switch:
                    if conf[key][1] == 'int':
                        interval = [int(param) for param in conf[key][2]]
                    elif conf[key][1] == 'float':
                        interval = [float(param) for param in conf[key][2]]
                    elif conf[key][1] in ['str', 'string']:
                        interval = [str(param) for param in conf[key][2]]
                    else:
                        interval = conf[key][2]
                    conf[key] = switch[conf[key][0]](interval)
                else:
                    conf[key] = conf[key]
        elif isinstance(conf[key], dict):
            add_hyperparameters(conf[key])
    return conf

def create_config(config):  

    alg_config = config.algorithm_config 
    alg_config = add_hyperparameters(alg_config)

    return alg_config