import sys
sys.path.append('/Users/macintosh/Desktop/Experiment/pybullet_envs')
sys.path.append('/Users/macintosh/Desktop/Experiment/railrl-private')
sys.path.append('/Users/macintosh/Desktop/Experiment/multi_person_bc')



# Before running each experiment:
# change datapath, eval_person_id, run_id

import rlkit.util.hyperparameter as hyp
from launchers.bc_exp_launcher import bc_exp_launcher
path_func = lambda x: '/Users/macintosh/Desktop/Experiment/datasets/{0}.npy'.format(x)

variant = dict(
    env_name='VALMultiobjTray-v0',
    env_kwargs={'test_env': True, 'gui': True},
    horizon=65,
    
    model_kwargs={},
    datapath=path_func('tray_demos_person_2'),

    use_gpu=False,
    log_dir='/Users/macintosh/Desktop/Experiment/images/',
    eval_person_id=2
    )

if __name__ == "__main__":
    search_space = {
        "seed": range(1),
        "demo_size": [500],
        "batch_size": [32],
        "weight_decay": [0.0],
        # "eval_person_id": range(1)
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(search_space, default_parameters=variant,)

    exp_id = 0
    for variant in sweeper.iterate_hyperparameters():

        bc_exp_launcher(variant, run_id=300, exp_id=exp_id)
        exp_id += 1