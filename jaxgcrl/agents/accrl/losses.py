import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) -
            jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config, networks, transitions, training_state, key, action_grad_gamma=0.0):
    def get_aux_metrics(actor_params, transitions):
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        state = transitions.state
        observation = jnp.concatenate([state, goal], axis=1)

        def f(input):
            means, _ = networks["actor"].apply(actor_params, input)
            action_magnitudes = jnp.linalg.norm(means, axis=-1)  # shape = (batch_size, action_chunk_len)
            action_magnitudes = jnp.mean(action_magnitudes, axis=0)  # shape = (action_chunk_len,)
            loss = jnp.mean(means, axis=0)  # shape = (action_chunk_len, action_dim)

            return loss, action_magnitudes

        jac, action_magnitudes = jax.jacrev(f, has_aux=True)(observation)

        aux_metrics = {}
        for i in range(action_magnitudes.shape[0]):
            aux_metrics[f"action{i}/magnitude"] = action_magnitudes[i]
            action_jac = jac[i]  # shape = (action_dim, batch_size, observation_dim)
            grad_norm = jnp.mean(jnp.linalg.norm(action_jac, axis=-1))
            aux_metrics[f"action{i}/grad_wrt_input_norm"] = grad_norm

        return aux_metrics

    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        state = transitions.state
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        og_shape = means.shape
        og_ndim = means.ndim

        means = jnp.reshape(means, (means.shape[0], -1))
        log_stds = jnp.reshape(log_stds, (log_stds.shape[0], -1))
        stds = jnp.exp(log_stds)
        x_ts = means + stds * \
            jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)  # dimension = B

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        if action_grad_gamma > 0.0 and og_ndim > 2:
            action_chunks = jnp.reshape(action, og_shape)
            chunk_len = action_chunks.shape[1]
            qf_pi = 0.0
            gamma = 1.0
            for i in range(chunk_len):
                actions_before = jax.lax.stop_gradient(action_chunks[:, :i, :])
                action_itself = action_chunks[:, i:i+1, :]
                actions_after = jax.lax.stop_gradient(action_chunks[:, i+1:, :])
                action_aux = jnp.concatenate([actions_before, action_itself, actions_after], axis=1)
                action_aux = jnp.reshape(action_aux, (action_aux.shape[0], -1))
                sa_repr = networks["sa_encoder"].apply(
                        sa_encoder_params, jnp.concatenate([state, action_aux], axis=-1))
                qf_pi += gamma * energy_fn(config["energy_fn"], sa_repr, g_repr)
                gamma *= action_grad_gamma
        else:
            sa_repr = networks["sa_encoder"].apply(
                sa_encoder_params, jnp.concatenate([state, action], axis=-1))

            qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = -alpha * \
            jnp.mean(jax.lax.stop_gradient(log_prob + config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )

    actor_grad_norm = optax.global_norm(actor_grad)
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params, log_prob)

    alpha_grad_norm = optax.global_norm(alpha_grad)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "actor_grad_norm": actor_grad_norm,
        "alpha_grad_norm": alpha_grad_norm,
    }
    # aux_metrics = get_aux_metrics(training_state.actor_state.params, transitions)
    # metrics.update(aux_metrics)

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.state
        goal = transitions.goal
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, goal
        )

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)

    critic_grad_norm = optax.global_norm(grad)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
        "critic_grad_norm": critic_grad_norm,
    }

    return training_state, metrics


def crl_action_sensitivity_metrics(
        energy_fn_name,
        networks,
        critic_params,
        crl_transitions,
        key,
        n_state_samples=128,
        n_action_samples=256
):
    """
    Measure variance and mean of Q-values over random actions for a fixed (s, g).

    Samples n_samples random actions uniformly in [-1, 1]^action_dim, and returns var/mean of Q-values.
    Used in CRL and ACCRL.
    """
    ind_key, action_key = jax.random.split(key)

    state = crl_transitions.state
    goal = crl_transitions.goal
    state = jnp.reshape(state, (-1, state.shape[-1]))
    goal = jnp.reshape(goal, (-1, goal.shape[-1]))

    batch_dim = state.shape[0]
    inds = jax.random.choice(ind_key, batch_dim, shape=(
        n_state_samples,), replace=False)
    state = state[inds]  # (n_state_samples, state_dim)
    goal = goal[inds]  # (n_state_samples, goal_dim)

    state = jnp.repeat(state[None, ...], n_action_samples, axis=0)
    goal = jnp.repeat(goal[None, ...], n_action_samples, axis=0)

    action_dim = crl_transitions.action.shape[-1]
    random_actions = jax.random.uniform(action_key, shape=(
        n_action_samples, action_dim), minval=-1.0, maxval=1.0)
    random_actions = jnp.repeat(random_actions[:, None, :], n_state_samples, axis=1)

    state_action = jnp.concatenate([state, random_actions], axis=-1)

    sa_encoder_params, g_encoder_params = (
        critic_params["sa_encoder"],
        critic_params["g_encoder"],
    )

    sa_repr = networks["sa_encoder"].apply(sa_encoder_params, state_action)
    g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

    # (n_action_samples, n_state_samples)
    q_values = energy_fn(energy_fn_name, sa_repr, g_repr)

    var_q = jnp.mean(jnp.var(q_values, axis=0))
    mean_q = jnp.mean(q_values)

    return {
        "critic/action_sensitivity_var_q": var_q,
        "critic/action_sensitivity_mean_q": mean_q,
    }
