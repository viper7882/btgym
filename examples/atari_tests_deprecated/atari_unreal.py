import os

from btgym_tf2.algorithms import AtariRescale42x42, Launcher, BaseAacPolicy, Unreal
cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=4,  # Set according CPU's available
    num_ps=1,
    num_envs=4,  # Number of invironments to run for every worker
    log_dir=os.path.expanduser('~/tmp/test_gym_unreal'),
)

env_config = dict(
    class_ref=AtariRescale42x42,  # Gym env. preprocessed to normalized grayscale 42x42 pix.
    kwargs={'gym_id': 'Breakout-v0'}
)

policy_config = dict(
    class_ref=BaseAacPolicy,
    kwargs={}
)
trainer_config = dict(
    class_ref=Unreal,
    kwargs=dict(
        opt_learn_rate=[1e-4, 1e-4], # Random log-uniform
        opt_end_learn_rate=1e-4,
        opt_decay_steps=50*10**6,
        model_gae_lambda=0.95,
        model_beta=[0.02, 0.001], # Entropy reg, random log-uniform
        rollout_length=20,
        time_flat=False,
        #replay_rollout_length=20,
        #replay_batch_size=1,
        use_off_policy_aac=False,
        use_reward_prediction=True,
        use_pixel_control=True,
        use_value_replay=True,
        vr_lambda=[1.0, 0.5],
        pc_lambda=[1.0, 0.1],
        rp_lambda=[1.0, 0.1],
        model_summary_freq=100,
        episode_summary_freq=10,
        env_render_freq=100,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=True,
    max_env_steps=50*10**6,
    root_random_seed=0,
    verbose=1
)
launcher.run()