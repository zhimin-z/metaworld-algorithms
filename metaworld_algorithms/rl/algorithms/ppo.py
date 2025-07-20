from dataclasses import dataclass
from typing import Literal, Self, override
from functools import partial

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import numpy.typing as npt
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.monitoring.utils import (
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from metaworld_algorithms.rl.networks import ContinuousActionPolicy, ValueFunction
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    Observation,
    Rollout,
    Value,
)

from .utils import LinearFeatureBaseline, compute_gae_scan, explained_variance, accumulate_concatenated_metrics
from .base import OnPolicyAlgorithm


@jax.jit
def _sample_action(
    policy: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _get_value(value_function: TrainState, observation: Observation) -> Value:
    return value_function.apply_fn(value_function.params, observation)


@jax.jit
def _eval_action(
    policy: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    return dist.mode()


@jax.jit
def _sample_action_dist_and_value(
    policy: TrainState,
    value_function: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    Value,
    PRNGKeyArray,
]:
    dist: distrax.Distribution
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    value = value_function.apply_fn(value_function.params, observation)
    return action, action_log_prob, dist.mode(), dist.stddev(), value, key  # pyright: ignore[reportReturnType]


@jax.jit
def _sample_action_dist(
    policy: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    PRNGKeyArray,
]:
    dist: distrax.Distribution
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    return action, action_log_prob, dist.mode(), dist.stddev(), key  # pyright: ignore[reportReturnType]


@dataclass(frozen=True)
class PPOConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    vf_config: ValueFunctionConfig | None = ValueFunctionConfig()
    clip_eps: float = 0.2
    baseline_type: Literal["linear", "mlp"] = "mlp"
    clip_vf_loss: bool = True
    entropy_coefficient: float = 5e-3
    vf_coefficient: float = 0.001
    normalize_advantages: bool = True
    gae_lambda: float = 0.97
    num_gradient_steps: int = 32
    num_epochs: int = 16
    target_kl: float | None = None


class PPO(OnPolicyAlgorithm[PPOConfig]):
    policy: TrainState
    value_function: TrainState | None
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    baseline_type: Literal["linear", "mlp"] = struct.field(pytree_node=False)
    clip_vf_loss: bool = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    vf_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)

    gae_lambda: float = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    target_kl: float | None = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(config: PPOConfig, env_config: EnvConfig, seed: int = 1) -> "PPO":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, vf_init_key = jax.random.split(master_key, 3)
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )

        policy_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.policy_config
        )
        policy = TrainState.create(
            apply_fn=policy_net.apply,
            params=policy_net.init(actor_init_key, dummy_obs),
            tx=config.policy_config.network_config.optimizer.spawn(),
        )

        value_function = None
        if config.vf_config is not None:
            assert config.baseline_type == "mlp", (
                "MLP baseline must be specified if vf_config is provided"
            )
            vf_net = ValueFunction(config.vf_config)
            value_function = TrainState.create(
                apply_fn=vf_net.apply,
                params=vf_net.init(vf_init_key, dummy_obs),
                tx=config.vf_config.network_config.optimizer.spawn(),
            )

        return PPO(
            num_tasks=config.num_tasks,
            policy=policy,
            value_function=value_function,
            key=algorithm_key,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            baseline_type=config.baseline_type,
            clip_vf_loss=config.clip_vf_loss,
            entropy_coefficient=config.entropy_coefficient,
            vf_coefficient=config.vf_coefficient,
            normalize_advantages=config.normalize_advantages,
            gae_lambda=config.gae_lambda,
            num_gradient_steps=config.num_gradient_steps,
            num_epochs=config.num_epochs,
            target_kl=config.target_kl,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        ret = {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ),
        }
        if self.baseline_type == "mlp":
            assert self.value_function is not None
            ret["vf_num_params"] = sum(
                x.size for x in jax.tree.leaves(self.value_function.params)
            )
        return ret

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.policy, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        if self.baseline_type == "mlp":
            action, log_prob, mean, std, value, key = _sample_action_dist_and_value(
                self.policy, self.value_function, observation, self.key
            )
            action, log_prob, mean, std, value = jax.device_get(
                (action, log_prob, mean, std, value)
            )
            aux_outputs = {
                "log_prob": log_prob,
                "mean": mean,
                "std": std,
                "value": value,
            }
        else:
            action, log_prob, mean, std, key = _sample_action_dist(
                self.policy, observation, self.key
            )
            action, log_prob, mean, std = jax.device_get((action, log_prob, mean, std))
            aux_outputs = {
                "log_prob": log_prob,
                "mean": mean,
                "std": std,
            }
        return (
            self.replace(key=key),
            action,
            aux_outputs,
        )

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.policy, observations))

    @jax.jit
    def _get_activations(
        self, data: Rollout, key: PRNGKeyArray, batch_size: int = 4096
    ) -> tuple[PyTree[Array], PyTree[Array] | None, PRNGKeyArray]:
        key, permutation_key = jax.random.split(key)
        rollout_size = data.observations.shape[0] * data.observations.shape[1]

        indices = jax.random.choice(permutation_key, rollout_size, shape=(batch_size,), replace=False)
        flattened_data = jax.tree.map(
            lambda x: x.reshape((rollout_size, *x.shape[2:])), data
        )
        activations_batch = flattened_data.observations[indices]

        _, policy_state = self.policy.apply_fn(
            self.policy.params, activations_batch, mutable=["intermediates"]
        )
        policy_acts = policy_state["intermediates"]

        vf_acts = None
        if self.baseline_type == "mlp":
            assert self.value_function is not None
            _, vf_state = self.value_function.apply_fn(
                self.value_function.params, activations_batch, mutable=["intermediates"]
            )
            vf_acts = vf_state["intermediates"]

        return policy_acts, vf_acts, key

    @partial(jax.jit, donate_argnames=("self"))
    def _update_inner(
        self,
        data: Rollout,
        next_obs: Observation
    ) -> tuple[Self, LogDict]:
        if self.baseline_type == "linear":
            # TODO: LinearFeatureBaseline in JAX
            # values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
            #     data, self.gamma
            # )
            # data = data._replace(values=values, returns=returns)
            # last_values = jnp.zeros(data.rewards.shape[1:], dtype=data.rewards.dtype)
            raise NotImplementedError("LinearFeatureBaseline not implemented in JAX")
        else:
            assert self.value_function is not None
            last_values = self.value_function.apply_fn(self.value_function.params, next_obs)

        data = compute_gae_scan(data, last_values, self.gamma, self.gae_lambda)

        assert data.advantages is not None and data.returns is not None
        assert data.values is not None and data.log_probs is not None
        diagnostic_logs = prefix_dict(
            "data",
            {
                **get_logs("advantages", data.advantages),
                **get_logs("returns", data.returns),
                **get_logs("values", data.values),
                **get_logs("rewards", data.rewards),
                **get_logs("actions", data.actions),
                **get_logs("num_episodes", data.dones.sum(axis=1), hist=False, std=False),
                "approx_entropy": -data.log_probs.mean(),
            },
        )

        def update_policy(
            policy: TrainState,
            data: Rollout,
            key: PRNGKeyArray,
        ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
            assert data.advantages is not None

            if self.normalize_advantages:
                advantages = (
                    data.advantages - data.advantages.mean(axis=0, keepdims=True)
                ) / (data.advantages.std(axis=0, keepdims=True) + 1e-8)
            else:
                advantages = data.advantages

            def policy_loss(params: FrozenDict):
                action_dist: distrax.Distribution
                new_log_probs: Float[Array, " *batch"]
                assert data.log_probs is not None

                action_dist = policy.apply_fn(
                    params,
                    data.observations,
                )
                new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
                log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
                ratio = jnp.exp(log_ratio)

                # For logs
                approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
                clip_fracs = jax.lax.stop_gradient(
                    (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
                )

                pg_loss1 = -advantages * ratio  # pyright: ignore[reportOptionalOperand]
                pg_loss2 = -advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                entropy_loss = action_dist.entropy().mean()

                return pg_loss - self.entropy_coefficient * entropy_loss, {
                    "metrics/entropy_loss": entropy_loss,
                    "metrics/policy_loss": pg_loss,
                    "metrics/approx_kl": approx_kl,
                    "metrics/clip_fracs": clip_fracs,
                }

            (_, logs), policy_grads = jax.value_and_grad(
                policy_loss, has_aux=True
            )(policy.params)
            policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
            grads_hist_dict = pytree_histogram(policy_grads["params"])

            policy = policy.apply_gradients(grads=policy_grads)
            policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params["params"])
            param_hist_dict = pytree_histogram(policy.params["params"])

            return (
                policy,
                key,
                logs
                | {
                    "nn/policy_gradient_norm": jnp.linalg.norm(policy_grads_flat),
                    "nn/policy_parameter_norm": jnp.linalg.norm(policy_params_flat),
                    **prefix_dict("nn/policy_gradients", grads_hist_dict),
                    **prefix_dict("nn/policy_parameters", param_hist_dict),
                },
            )

        def update_value_function(
            vf: TrainState,
            data: Rollout,
            key: PRNGKeyArray,
        ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
            def value_function_loss(params: FrozenDict):
                new_values: Float[Array, "*batch 1"]
                new_values = vf.apply_fn(
                    params,
                    data.observations,
                )

                chex.assert_equal_shape((new_values, data.returns))
                vf_loss = 0.5 * ((new_values - data.returns) ** 2).mean()

                return self.vf_coefficient * vf_loss, {
                        "metrics/vf_loss": vf_loss,
                        "metrics/values": new_values.mean(),
                    }


            (_, logs), vf_grads = jax.value_and_grad(
                value_function_loss, has_aux=True
            )(vf.params)
            vf_grads_flat, _ = jax.flatten_util.ravel_pytree(vf_grads)
            grads_hist_dict = pytree_histogram(vf_grads["params"])

            vf = vf.apply_gradients(grads=vf_grads)
            vf_params_flat, _ = jax.flatten_util.ravel_pytree(vf.params)
            param_hist_dict = pytree_histogram(vf.params["params"])

            return (
                vf,
                key,
                logs
                | {
                    "nn/vf_gradient_norm": jnp.linalg.norm(vf_grads_flat),
                    "nn/vf_parameter_norm": jnp.linalg.norm(vf_params_flat),
                    **prefix_dict("nn/vf_gradients", grads_hist_dict),
                    **prefix_dict("nn/vf_parameters", param_hist_dict),
                },
            )

        def train_minibatch(carry, minibatch: Rollout):
            policy, vf, key = carry
            vf_logs = {}
            policy, key, policy_logs = update_policy(policy, minibatch, key)

            if self.baseline_type == "mlp":
                vf, key, vf_logs = update_value_function(vf, minibatch, key)

            return (policy, vf, key), (policy_logs | vf_logs)

        def train_epoch(carry, _):
            policy, vf, key, data = carry

            key, permutation_key = jax.random.split(key)
            rollout_size = data.observations.shape[0] * data.observations.shape[1]

            permutation = jax.random.permutation(permutation_key, rollout_size)
            minibatched_data = jax.tree.map(
                lambda x: x.reshape((rollout_size, *x.shape[2:])), data
            )
            shuffled_data = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), minibatched_data
            )

            minibatches = jax.tree.map(
                lambda x: x.reshape(self.num_gradient_steps, -1, *x.shape[1:]),
                shuffled_data,
            )

            (policy, vf, key), logs = jax.lax.scan(
                train_minibatch, (policy, vf, key), minibatches
            )
            return (policy, vf, key, data), logs

        (policy, vf, key, _), logs = jax.lax.scan(
            train_epoch,
            (self.policy, self.value_function, self.key, data),
            None,
            length=self.num_epochs,
        )

        # Finalize logs
        final_logs = {}
        final_logs["metrics/explained_variance"] = explained_variance(
            data.values.reshape(-1), data.returns.reshape(-1)
        )
        final_logs.update(accumulate_concatenated_metrics(logs))

        return self.replace(key=key, value_function=vf, policy=policy), diagnostic_logs | final_logs

    @override
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]:
        del dones

        assert self.value_function is not None
        assert next_obs is not None
        self, logs = self._update_inner(data, next_obs)

        # log activations
        policy_acts, vf_acts, key = self._get_activations(data, self.key)
        logs.update(prefix_dict("nn/activations", pytree_histogram(policy_acts)))
        if vf_acts is not None:
            logs.update(prefix_dict("nn/vf_activations", pytree_histogram(vf_acts)))

        return self.replace(key=key), logs
