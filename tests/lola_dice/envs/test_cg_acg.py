import random

from lola_dice.envs import CG, AsymCG


def init_env(max_steps, batch_size, env_class, seed=1):
    env = env_class(max_steps, batch_size)
    env.seed(seed)
    return env


def check_obs(obs, batch_size, grid_size):
    assert len(obs) == 2, "two players"
    for i in range(batch_size):
        for player_obs in obs:
            assert player_obs.shape == (batch_size, 4, grid_size, grid_size)
            assert player_obs[i, 0, ...].sum() == 1.0, f"observe 1 player red in grid: {player_obs[i, 0, ...]}"
            assert player_obs[i, 1, ...].sum() == 1.0, f"observe 1 player blue in grid: {player_obs[i, 1, ...]}"
            assert player_obs[i, 2:, ...].sum() == 1.0, f"observe 1 coin in grid: {player_obs[i, 0, ...]}"


def assert_logger_buffer_size(env, n_steps):
    assert len(env.coin_pick_speed) == n_steps
    assert len(env.rewards_red) == n_steps
    assert len(env.rewards_blue) == n_steps
    assert len(env.player_blue_picked_own) == n_steps
    assert len(env.player_red_picked_own) == n_steps
    assert len(env.player_blue_picked) == n_steps
    assert len(env.player_red_picked) == n_steps


def test_reset():
    max_steps, batch_size, grid_size = 20, 5, 3
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env in [coin_game, asymm_coin_game]:
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)


def test_step():
    max_steps, batch_size, grid_size = 20, 5, 3
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env in [coin_game, asymm_coin_game]:
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        actions = [[random.randint(0, env.NUM_ACTIONS - 1) for b in range(batch_size)] for p in range(env.NUM_AGENTS)]
        obs, reward, done, info = env.step(actions)
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=1)
        assert not done


def test_multiple_steps():
    max_steps, batch_size, grid_size = 20, 5, 3
    n_steps = int(max_steps * 0.75)
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env in [coin_game, asymm_coin_game]:
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        for step_i in range(1, n_steps, 1):
            actions = [[random.randint(0, env.NUM_ACTIONS - 1) for b in range(batch_size)] for p in
                       range(env.NUM_AGENTS)]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done


def test_multiple_episodes():
    max_steps, batch_size, grid_size = 20, 100, 3
    n_steps = int(max_steps * 8.25)
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env in [coin_game, asymm_coin_game]:
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[random.randint(0, env.NUM_ACTIONS - 1) for b in range(batch_size)] for p in
                       range(env.NUM_AGENTS)]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0


def overwritte_pos(batch_size, env, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos):
    assert c_red_pos is None or c_blue_pos is None
    if c_red_pos is None:
        env.red_coin = [False] * batch_size
        coin_pos = c_blue_pos
    if c_blue_pos is None:
        env.red_coin = [True] * batch_size
        coin_pos = c_red_pos

    env.red_pos = [p_red_pos] * batch_size
    env.blue_pos = [p_blue_pos] * batch_size
    env.coin_pos = [coin_pos] * batch_size


def test_logged_info_no_picking():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env in [coin_game, asymm_coin_game]:
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 0
                assert info["rewards_red"] == 0
                assert info["rewards_blue"] == 0
                assert "pick_own_red" not in info.keys()
                assert "pick_own_blue" not in info.keys()
                assert info["pick_speed_red"] == 0
                assert info["pick_speed_blue"] == 0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == 1.0
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == 2.0
                assert info["rewards_blue"] == 0
                assert "pick_own_red" not in info.keys()
                assert "pick_own_blue" not in info.keys()
                assert info["pick_speed_red"] == 1.0
                assert info["pick_speed_blue"] == 0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == -2.0
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == -1.0
                assert info["rewards_blue"] == 1.0
                assert "pick_own_red" not in info.keys()
                assert "pick_own_blue" not in info.keys()
                assert info["pick_speed_red"] == 0
                assert info["pick_speed_blue"] == 1.0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                assert info["rewards_red"] == 0
                assert info["rewards_blue"] == 1.0
                assert "pick_own_red" not in info.keys()
                assert "pick_own_blue" not in info.keys()
                assert info["pick_speed_red"] == 0
                assert info["pick_speed_blue"] == 1.0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                assert info["rewards_red"] == 1.0
                assert info["rewards_blue"] == -2.0
                assert "pick_own_red" not in info.keys()
                assert "pick_own_blue" not in info.keys()
                assert info["pick_speed_red"] == 1.0
                assert info["pick_speed_blue"] == 0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                assert info["rewards_red"] == 1.0
                assert info["rewards_blue"] == -1.0
                assert info["pick_own_red"] == 0.0
                assert info["pick_own_blue"] == 1.0
                assert info["pick_speed_red"] == 1.0
                assert info["pick_speed_blue"] == 1.0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == -1.0
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == 1.0
                assert info["rewards_blue"] == 1.0
                assert info["pick_own_red"] == 1.0
                assert info["pick_own_blue"] == 0.0
                assert info["pick_speed_red"] == 1.0
                assert info["pick_speed_blue"] == 1.0

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])




def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == -0.5
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == 0.5
                assert info["rewards_blue"] == 0.5
                assert info["pick_own_red"] == 1.0
                assert info["pick_own_blue"] == 0.0
                assert info["pick_speed_red"] == 0.5
                assert info["pick_speed_blue"] == 0.5

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])



def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                assert info["rewards_red"] == 0.5
                assert info["rewards_blue"] == -0.5
                assert info["pick_own_red"] == 0.0
                assert info["pick_own_blue"] == 1.0
                assert info["pick_speed_red"] == 0.5
                assert info["pick_speed_blue"] == 0.5

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])



def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == -0.5
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == 0.5
                assert info["rewards_blue"] == 0.5
                assert info["pick_own_red"] == 1.0
                assert info["pick_own_blue"] == 0.0
                assert info["pick_speed_red"] == 0.5
                assert info["pick_speed_blue"] == 0.5

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])



def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 0.75
                assert info["rewards_red"] == 0.25
                assert info["rewards_blue"] == 0.0
                assert info["pick_own_red"] == 0.0
                assert info["pick_own_blue"] == 1.0
                assert info["pick_speed_red"] == 0.25
                assert info["pick_speed_blue"] == 0.5

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])



def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [1, 1], None]
    c_blue_pos = [None, [1, 1], None, [1, 1]]
    max_steps, batch_size, grid_size = 4, 28, 3
    n_steps = max_steps
    coin_game = init_env(max_steps, batch_size, CG)
    asymm_coin_game = init_env(max_steps, batch_size, AsymCG)

    for env_i, env in enumerate([coin_game, asymm_coin_game]):
        obs, info = env.reset()
        check_obs(obs, batch_size, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwritte_pos(batch_size, env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])

        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = [[p_red_act[step_i - 1]] * batch_size,
                       [p_blue_act[step_i - 1]] * batch_size]
            obs, reward, done, info = env.step(actions)
            check_obs(obs, batch_size, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done or (step_i == max_steps and done)
            if done:
                assert info["pick_speed"] == 1.0
                if env_i == 0:  # coin game
                    assert info["rewards_red"] == 0.0
                elif env_i == 1:  # asymmetric coin game
                    assert info["rewards_red"] == 2.0/4
                assert info["rewards_blue"] == 0.0
                assert info["pick_own_red"] == 0.5
                assert info["pick_own_blue"] == 0.5
                assert info["pick_speed_red"] == 0.5
                assert info["pick_speed_blue"] == 0.5

                obs, info = env.reset()
                check_obs(obs, batch_size, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

            overwritte_pos(batch_size, env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i],
                           c_blue_pos[step_i])
