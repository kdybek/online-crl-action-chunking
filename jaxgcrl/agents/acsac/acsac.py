# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import logging
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.agents.sac import losses as sac_losses
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, Policy, PRNGKey
from brax import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ChunkedActorEvaluator, Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from types import SimpleNamespace

from . import networks

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class ChunkedTransition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    valid: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class EvalDummyTrainingState(NamedTuple):
    """Dummy training state for evaluation."""

    actor_state: Any


def get_actions(
    make_policy: Policy,
    params: Params,
    observation: NestedArray,
    action_size: int,
    action_chunk_length: int,
    key: PRNGKey,
):
    """Get actions from the policy."""
    policy = make_policy(params)
    flat_action, policy_extras = policy(observation, key)

    assert flat_action.shape == (observation.shape[0], action_size * action_chunk_length)

    # Reshape the flat action to have shape (B, action_chunk_length, action_size)
    return jnp.reshape(flat_action, (-1, action_chunk_length, action_size))


def env_step(
    action,
    env: Env,
    env_state: State,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    nstate = env.step(env_state, action)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=action,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"state_extras": state_extras},
    )


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


@functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_batch(config, env, transition: Transition, sample_key: PRNGKey) -> ChunkedTransition:
    if config.use_her:
        # Find truncation indexes if present
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )

        # final_step_mask.shape == (seq_len, seq_len)
        final_step_mask = (
            is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        )
        final_step_mask = jnp.logical_and(
            final_step_mask,
            transition.extras["state_extras"]["truncation"][None, :],
        )
        non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

        # If final state is not present use original goal (i.e. don't change anything)
        new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
        binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

        new_goals = (
            binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
            + jnp.logical_not(binary_mask)[:, None]
            * transition.observation[new_goals_idx][:, env.state_dim :]
        )

        # Transform observation
        state = transition.observation[:, : env.state_dim]
        new_obs = jnp.concatenate([state, new_goals], axis=1)

        # Recalculate reward
        dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
        new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)

        # Transform next observation
        next_state = transition.next_observation[:, : env.state_dim]
        new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

        transition = transition._replace(
            observation=jnp.squeeze(new_obs),
            next_observation=jnp.squeeze(new_next_obs),
            reward=jnp.squeeze(new_reward),
        )

    # --- Gather action chunks for critic update ---
    seq_len, action_size = transition.action.shape

    # Indices for gathering action chunks
    base_idx = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
    offsets = jnp.arange(config.action_chunk_length)[None, :]  # (1, action_chunk_length)
    idx = base_idx + offsets  # (seq_len, action_chunk_length)

    # Mask for valid indices (inside sequence)
    valid_mask = idx < seq_len

    # Clip indices to avoid OOB access
    idx_clipped = jnp.clip(idx, 0, seq_len - 1)

    # Gather actions
    gathered_actions = transition.action[idx_clipped]  # (seq_len, action_chunk_length, action_size)
    gathered_rewards = transition.reward[idx_clipped]  # (seq_len, action_chunk_length)
    next_observation = transition.next_observation[idx_clipped[:, -1]]
    discount = transition.discount[idx_clipped[:, -1]]

    # Gather traj_ids
    gathered_traj_ids = transition.extras["state_extras"]["traj_id"][idx_clipped]  # (seq_len, action_chunk_length)

    # Compare traj_ids with the first one in each chunk
    first_traj_ids = transition.extras["state_extras"]["traj_id"][:, None]  # (seq_len, 1)
    same_traj_mask = gathered_traj_ids == first_traj_ids  # (seq_len, action_chunk_length)

    mask = valid_mask & same_traj_mask  # (seq_len, action_chunk_length)
    valid = jnp.all(mask, axis=-1)  # (seq_len,)

    gathered_actions = jnp.where(
        mask[..., None],
        gathered_actions,
        jnp.zeros_like(gathered_actions),
    )
    gathered_rewards = jnp.where(mask, gathered_rewards, jnp.zeros_like(gathered_rewards))
    discounted_rewards = jnp.sum(gathered_rewards * (config.discounting ** offsets), axis=-1)

    flat_action_chunk = jnp.reshape(gathered_actions, (seq_len, -1))  # (seq_len, action_chunk_length * action_size)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"]),
        }
    }

    return ChunkedTransition(
        observation=transition.observation,
        next_observation=next_observation,
        action=flat_action_chunk,
        reward=discounted_rewards,
        discount=discount,
        extras=extras,
        valid=valid,
    )


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class ACSAC:
    """Action Chunked Soft Actor-Critic (ACSAC) agent."""

    learning_rate: float = 3e-4
    discounting: float = 0.99
    batch_size: int = 256
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    # target update rate
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    # layer norm
    use_ln: bool = False
    # hindsight experience replay
    use_her: bool = True

    action_chunk_length: int = 3

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use,
            device_count,
        )
        network_factory: types.NetworkFactory[networks.SACNetworks] = networks.make_sac_networks

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError("No training will happen because min_replay_size >= total_env_steps")

        if self.max_replay_size is None:
            max_replay_size = config.total_env_steps
        else:
            max_replay_size = self.max_replay_size

        # The number of environment steps executed for every `actor_step()` call.
        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        logging.info("Num_prefill_actor_steps: %s", num_prefill_actor_steps)
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of epoch calls per training
        # equals to
        # ceil(config.total_env_steps - num_prefill_env_steps /
        #      (num_evals_after_init * env_steps_per_actor_step))
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        assert config.num_envs % device_count == 0
        env = train_env
        if isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(key, config.num_envs // jax.process_count() // local_devices_to_use),
            )
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        obs_size = env.observation_size
        action_size = env.action_size
        action_chunk_size = action_size * self.action_chunk_length

        def normalize_fn(x, y):
            return x

        if self.normalize_observations:
            normalize_fn = running_statistics.normalize
        sac_network = network_factory(
            observation_size=obs_size,
            action_size=action_chunk_size,
            preprocess_observations_fn=normalize_fn,
            layer_norm=self.use_ln,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
        )
        make_policy = networks.make_inference_fn(sac_network)

        alpha_optimizer = optax.adam(learning_rate=3e-4)

        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=dummy_obs,
            next_observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                },
                "policy_extras": {},
            },
        )
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )

        alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
            sac_network=sac_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting ** self.action_chunk_length,
            action_size=action_chunk_size,
        )
        alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )

        def update_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                training_state.policy_params,
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
            alpha = jnp.exp(training_state.alpha_params)
            critic_loss, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )
            actor_loss, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )

            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                q_params,
            )

            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": jnp.exp(alpha_params),
            }

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
                normalizer_params=training_state.normalizer_params,
            )
            return (new_training_state, key), metrics

        def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params: Params,
            env_state: Union[envs.State, envs_v1.State],
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[
            running_statistics.RunningStatisticsState,
            Union[envs.State, envs_v1.State],
            ReplayBufferState,
        ]:
            @jax.jit
            def f(carry, unused_t):
                state, current_key, chunk_idx, actions = carry

                action_key, next_key = jax.random.split(current_key, 2)

                chunk_idx = jnp.where(state.info["is_first"], 0, chunk_idx)
                need_replan = chunk_idx == 0

                new_actions = get_actions(make_policy, (normalizer_params, policy_params), state.obs, action_size, self.action_chunk_length, action_key)

                actions = jnp.where(
                    need_replan[:, None, None],
                    new_actions,
                    actions,
                )

                action = actions[jnp.arange(actions.shape[0]), chunk_idx]

                nstate, transition = env_step(action, env, state, extra_fields=("truncation", "traj_id"))

                chunk_idx = (chunk_idx + 1) % self.action_chunk_length

                return (nstate, next_key, chunk_idx, actions), transition

            B = env_state.obs.shape[0]
            init_chunk_idx = jnp.zeros((B,), dtype=jnp.int32)

            actions = get_actions(make_policy, (normalizer_params, policy_params), env_state.obs, action_size, self.action_chunk_length, key)  # Not optimal, but should be fine for now.
            (env_state, _, _, _), data = jax.lax.scan(
                f,
                (env_state, key, init_chunk_idx, actions),
                (),
                length=self.unroll_length
            )

            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation,  # so that batch_size*unroll_length is the first dimension
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, buffer_state

        def training_step(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
            return training_state, env_state, buffer_state, metrics

        def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                new_normalizer_params, env_state, buffer_state = get_experience(
                    training_state.normalizer_params,
                    training_state.policy_params,
                    env_state,
                    buffer_state,
                    key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            experience_key, training_key, sampling_key, sanitization_key = jax.random.split(key, 4)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, None, 0, 0))(
                self, env, transitions, batch_keys
            )

            # Shuffle transitions and reshape them into (num_update_steps, batch_size, ...)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )

            valid = transitions.valid
            N = valid.shape[0]
            idx = jnp.arange(N)
            valid_idx = jnp.where(valid, idx, 0)  # replace invalid indices with 0
            valid_idx = jnp.sort(valid_idx)[::-1]  # put 0s at the end
            num_valid = jnp.maximum(jnp.sum(valid), 1)
            cycle_idx = idx % num_valid
            valid_idx = valid_idx[cycle_idx]

            replacement_idx = jax.random.choice(
                sanitization_key,
                valid_idx,
                shape=(N,),
                replace=True
            )

            final_idx = jnp.where(valid, idx, replacement_idx)

            transitions = jax.tree_util.tree_map(
                lambda x: x[final_idx],
                transitions
            )
            permutation = jax.random.permutation(experience_key, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, unsued_t):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, bs, metrics = training_step(ts, es, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(self.train_step_multiplier - 1, ts, bs, update_key)
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            nonlocal training_walltime
            t = time.time()
            (training_state, env_state, buffer_state, metrics) = training_epoch(
                training_state, env_state, buffer_state, key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                buffer_state,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            sac_network=sac_network,
            alpha_optimizer=alpha_optimizer,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        # Replay buffer init
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        if not eval_env:
            eval_env = env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
            )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        get_actions_func = functools.partial(
            get_actions,
            make_policy=make_policy,
            action_size=action_size,
            action_chunk_length=self.action_chunk_length,
        )
        evaluator_key, eval_key = jax.random.split(eval_key)
        evaluator = ChunkedActorEvaluator(
            get_actions=get_actions_func,
            action_step=env_step,
            receding_horizon=self.action_chunk_length,
            eval_env=eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=evaluator_key,
            full_chunk=True,
            need_actor_key=True,
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            eval_dummy_training_state = EvalDummyTrainingState(actor_state=_unpmap((training_state.normalizer_params, training_state.policy_params)))
            metrics = evaluator.run_evaluation(
                eval_dummy_training_state
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap((training_state.normalizer_params, training_state.policy_params)),
                unwrapped_env,
                do_render=False
            )

        # Create and initialize the replay buffer.
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_keys
        )

        replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        logging.info("replay size after prefill %s", replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info("step %s", current_step)

            # Optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # Eval and logging
            if process_id == 0:
                if config.checkpoint_logdir:
                    # Save current policy.
                    params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                    path = f"{config.checkpoint_logdir}_sac_{current_step}.pkl"
                    model.save_params(path, params)

                # Run evals.
                eval_dummy_training_state = EvalDummyTrainingState(actor_state=_unpmap((training_state.normalizer_params, training_state.policy_params)))
                metrics = evaluator.run_evaluation(
                    eval_dummy_training_state
                )
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    unwrapped_env,
                    do_render=False,
                )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
