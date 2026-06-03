import time

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import acting
from brax.training.types import Metrics, PolicyParams


# TODO: make only single type of Evaluator
class Evaluator(acting.Evaluator):
    """
    Class for running evaluation epochs in a training process.

    This class extends the base acting.Evaluator and is designed to evaluate the
    performance of policies based on given parameters and training metrics.
    It provides detailed evaluations by aggregating various metrics over episodes
    and calculating statistical summaries such as mean, standard deviation,
    maximum, and minimum values.

    This is an evaluator that behaves in the exact same way as brax Evaluator,
    but additionally it aggregates metrics with max, min.
    It also logs in how many episodes there was any success.
    """

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(
            training_state, env, state, extra_fields=extra_fields)
        return nstate, transition

    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data


def generate_chunked_unroll(get_actions, action_step, receding_horizon, training_state, env, env_state, unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        (state, chunk_idx, actions) = carry

        new_actions = get_actions(training_state.actor_state, state.obs)

        need_replan = chunk_idx == 0

        actions = jnp.where(
            need_replan[:, None, None],
            new_actions,
            actions
        )

        action = actions[jnp.arange(actions.shape[0]), chunk_idx, :]

        nstate, transition = action_step(action, env, state, extra_fields=extra_fields)

        chunk_idx = (chunk_idx + 1) % receding_horizon

        return (nstate, chunk_idx, actions), transition

    B = env_state.obs.shape[0]

    init_chunk_idx = jnp.zeros(B, dtype=jnp.int32)
    actions = get_actions(training_state.actor_state, env_state.obs)
    (final_state, _, _), data = jax.lax.scan(
        f, (env_state, init_chunk_idx, actions), (), length=unroll_length)
    return final_state, data


class ActorEvaluator:
    """Single GPU evaluator that evaluates an arbitrary actor function. Used by the CRL agent."""

    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            # (np.std, "_std"),
            # (np.max, "_max"),
            # (np.min, "_min"),
        ]

        wanted_metric_names = [
            "reward",
            "success",
            "success_easy",
            "dist",
            "distance_from_origin",
        ]

        available_metric_names = eval_metrics.episode_metrics.keys()

        metric_names = [
            name for name in wanted_metric_names if name in available_metric_names]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in metric_names
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics


class ChunkedActorEvaluator:
    """Single GPU evaluator that evaluates an arbitrary chunked actor function. Used by the ACCRL agent."""

    def __init__(self, get_actions, action_step, receding_horizon, eval_env, num_eval_envs, episode_length, key, full_chunk):
        self._key = key
        self._eval_walltime = 0.0
        self._receding_horizon = receding_horizon
        self._full_chunk = full_chunk

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_chunked_unroll(
                get_actions,
                action_step,
                receding_horizon,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            # (np.std, "_std"),
            # (np.max, "_max"),
            # (np.min, "_min"),
        ]

        wanted_metric_names = [
            "reward",
            "success",
            "success_easy",
            "dist",
            "distance_from_origin",
        ]

        available_metric_names = eval_metrics.episode_metrics.keys()

        metric_names = [
            name for name in wanted_metric_names if name in available_metric_names]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in metric_names
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0)

        metrics["avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["epoch_eval_time"] = epoch_eval_time
        metrics["sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics["walltime"] = self._eval_walltime

        metrics_rh = {
            f"eval/rh{self._receding_horizon}/{key}": value for key, value in metrics.items()}
        metrics_full_chunk = {}

        if self._full_chunk:
            metrics_full_chunk = {
                f"eval/full_chunk/{key}": value for key, value in metrics.items()}

        metrics_all = {**metrics_rh, **metrics_full_chunk}

        return metrics_all
