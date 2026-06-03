import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ChunkedActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_critic, crl_action_sensitivity_metrics
from .networks import Actor, Encoder

Metrics = types.Metrics
Env = envs.Env
State = envs.State


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


class CRLTransition(NamedTuple):
    """Container for a transition used in CRL updates"""
    state: jnp.ndarray
    goal: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch(buffer_config, transition, key):
    sample_key, shuffle_key = jax.random.split(key)
    gamma, state_size, goal_indices, action_chunk_length, shuffle_actions = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )

    # matrix of shape seq_len, seq_len where entry is 1 if the corresponding time indices have the same traj_id and 0 otherwise
    trajectory_mask = jnp.equal(single_trajectories, single_trajectories.T)

    probs = probs * trajectory_mask + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered

    # --- Gather action chunks for critic update ---
    seq_len, action_size = transition.action.shape

    # Indices for gathering action chunks
    base_idx = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
    offsets = jnp.arange(action_chunk_length)[None, :]  # (1, action_chunk_length)
    idx = base_idx + offsets  # (seq_len, action_chunk_length)

    # Mask for valid indices (inside sequence)
    valid_mask = idx < seq_len

    # Clip indices to avoid OOB access
    idx_clipped = jnp.clip(idx, 0, seq_len - 1)

    # Gather actions
    gathered_actions = transition.action[idx_clipped]  # (seq_len, action_chunk_length, action_size)

    # Gather traj_ids
    gathered_traj_ids = transition.extras["state_extras"]["traj_id"][idx_clipped]  # (seq_len, action_chunk_length)

    # Compare traj_ids with the first one in each chunk
    first_traj_ids = transition.extras["state_extras"]["traj_id"][:, None]  # (seq_len, 1)
    same_traj_mask = gathered_traj_ids == first_traj_ids  # (seq_len, action_chunk_length)

    mask = valid_mask & same_traj_mask  # (seq_len, action_chunk_length)

    gathered_actions = jnp.where(
        mask[..., None],
        gathered_actions,
        jnp.zeros_like(gathered_actions),
    )

    perm = jax.random.permutation(shuffle_key, action_chunk_length)
    gathered_actions = jax.lax.cond(
        shuffle_actions,
        lambda: jnp.take(gathered_actions, perm, axis=-2),
        lambda: gathered_actions,
    )

    flat_action_chunk = jnp.reshape(gathered_actions, (seq_len, -1))[:-1]  # (seq_len-1, action_chunk_length * action_size)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "future_state": future_state,
        "future_action": future_action,
    }

    return CRLTransition(
        state=state,
        goal=goal,
        action=flat_action_chunk,
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


@dataclass
class ACCRL:
    """Contrastive Reinforcement Learning with Action Chunking (ACCRL) agent."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    action_chunk_length: int = 3

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False
    target_entropy_coeff: float = 1.0

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 60
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    action_noise_std: float = 0.0

    random_replanning: bool = False
    action_shuffling: bool = False

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: envs.Env,
        eval_env: Optional[envs.Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = np.ceil(
            (config.total_env_steps - num_prefill_env_steps) / (config.num_evals * env_steps_per_actor_step)
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        action_chunk_size = action_size * self.action_chunk_length
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            action_chunk_length=self.action_chunk_length,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_chunk_size]))
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_chunk_size * self.target_entropy_coeff
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        logging.info("Number of actor parameters: %d", count_params(actor_state.params))
        logging.info("Number of critic parameters: %d", count_params(critic_state.params))
        logging.info("Number of alpha parameters: %d", count_params(alpha_state.params))

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        def get_actions(actor_state, obs, key):
            means, log_stds = actor.apply(actor_state.params, obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
            return actions

        def get_deterministic_actions(actor_state, obs):
            means, _ = actor.apply(actor_state.params, obs)
            actions = nn.tanh(means)
            return actions

        def action_step(action, env, env_state, extra_fields):
            nstate = env.step(env_state, action)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=action,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                state, current_key, chunk_idx, actions, replan_len = carry

                action_key, replan_key, noise_key, next_key = jax.random.split(current_key, 4)

                need_replan = chunk_idx == 0

                new_actions = get_actions(actor_state, state.obs, action_key)

                actions = jnp.where(
                    need_replan[:, None, None],
                    new_actions,
                    actions,
                )

                new_replan_len = jax.random.randint(
                    replan_key,
                    shape=(actions.shape[0],),
                    minval=1,
                    maxval=self.action_chunk_length + 1,
                )

                replan_len = jnp.where(
                    need_replan & self.random_replanning,
                    new_replan_len,
                    replan_len,
                )

                action = actions[jnp.arange(actions.shape[0]), chunk_idx]

                noise = jax.lax.cond(
                    self.action_noise_std > 0.0,
                    lambda: self.action_noise_std * jax.random.normal(noise_key, shape=action.shape),
                    lambda: jnp.zeros_like(action),
                )
                action = jnp.clip(action + noise, -1.0, 1.0)

                nstate, transition = action_step(action, train_env, state, extra_fields=("truncation", "traj_id"))

                done = nstate.done

                chunk_idx = jnp.where(done, 0, (chunk_idx + 1) % replan_len)

                return (nstate, next_key, chunk_idx, actions, replan_len), transition

            B = env_state.obs.shape[0]
            chunk_idx = jnp.zeros((B,), dtype=jnp.int32)
            replan_len = jnp.full((B,), self.action_chunk_length, dtype=jnp.int32)

            actions = get_actions(actor_state, env_state.obs, key)  # Not optimal, but should be fine for now.
            (env_state, _, _, _, _), data = jax.lax.scan(
                f,
                (env_state, key, chunk_idx, actions, replan_len),
                (),
                length=self.unroll_length
            )

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_state,
                    key,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics


        @jax.jit
        def process_transitions(transitions, key):
            sampling_key, permutation_key = jax.random.split(key)
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            crl_transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, train_env.goal_indices, self.action_chunk_length, self.action_shuffling),
                transitions,
                batch_keys,
            )
            crl_transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), crl_transitions
            )

            # permute transitions
            permutation = jax.random.permutation(permutation_key, len(crl_transitions.state))
            crl_transitions = jax.tree_util.tree_map(lambda x: x[permutation], crl_transitions)
            crl_transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                crl_transitions,
            )

            return crl_transitions

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key, process_key, training_key = jax.random.split(key, 3)

            # update buffer
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                env_state,
                buffer_state,
                experience_key,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # process transitions for training
            crl_transitions = process_transitions(transitions, process_key)

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), crl_transitions)

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    metrics,
                ) = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        @jax.jit
        def compute_action_sensitivity_metrics(buffer_state, critic_params, key):
            process_key, sensitivity_key = jax.random.split(key)
            _, transitions = replay_buffer.sample(buffer_state)
            crl_transitions = process_transitions(transitions, process_key)
            networks = dict(
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )
            sensitivity_metrics = crl_action_sensitivity_metrics(
                self.energy_fn,
                networks,
                critic_params,
                crl_transitions,
                sensitivity_key,
            )
            return sensitivity_metrics

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        receding_horizons = [1, 3, 5, 10, 15, 30]
        receding_horizons = [h for h in receding_horizons if h <= self.action_chunk_length]
        """Setting up evaluators"""
        evaluators = [
            ChunkedActorEvaluator(
                get_deterministic_actions,
                action_step,
                receding_horizon,
                eval_env,
                num_eval_envs=config.num_eval_envs,
                episode_length=config.episode_length,
                key=eval_env_key,
                full_chunk=receding_horizon == self.action_chunk_length,
            ) for receding_horizon in receding_horizons
        ]

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            for evaluator in evaluators:
                eval_metrics = evaluator.run_evaluation(training_state)
                metrics.update(eval_metrics)

            key, sensitivity_key = jax.random.split(key)
            sensitivity_metrics = compute_action_sensitivity_metrics(
                buffer_state,
                training_state.critic_state.params,
                sensitivity_key,
            )
            metrics.update(sensitivity_metrics)

            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
