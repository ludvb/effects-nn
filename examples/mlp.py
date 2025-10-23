from typing import Callable, NamedTuple

import effects_optim as optim
import jax
import jax.numpy as jnp

import effects_nn as nn


class LinearState(NamedTuple):
    weight: jax.Array
    bias: jax.Array


@nn.init_program
def linear_init(in_features: int, out_features: int):
    weight = jax.random.normal(jax.random.PRNGKey(0), (in_features, out_features))
    bias = jnp.zeros(out_features)
    return LinearState(weight=weight, bias=bias)


@nn.pure_program
def linear_apply(state: LinearState, x: jax.Array):
    print(nn.program_store.get_path(), state.weight)
    return jnp.dot(x, state.weight) + state.bias


class MLPState(NamedTuple):
    num_layers: int
    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu


@nn.init_program
def mlp_init(
    num_layers: int,
    in_features: int,
    hidden_features: int,
    out_features: int,
):
    for i in range(num_layers):
        in_features_cur = in_features if i == 0 else hidden_features
        out_features_cur = hidden_features if i < num_layers - 1 else out_features
        with nn.program_store.scope(f"layer_{i}"):
            linear_init(in_features_cur, out_features_cur)
    return MLPState(num_layers=num_layers)


@nn.pure_program
def mlp_apply(state: MLPState, x: jax.Array):
    for i in range(state.num_layers):
        with nn.program_store.scope(f"layer_{i}"):
            x = linear_apply(x)
        if i < state.num_layers - 1:
            x = state.activation_fn(x)
    return x


def main():
    with nn.program_store.handler():
        mlp_init(
            num_layers=3,
            in_features=4,
            hidden_features=8,
            out_features=2,
        )
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        output = mlp_apply(x)
    print("Impure output:", output)

    mlp_init_pure = nn.unlift_program(mlp_init)
    mlp_apply_pure = nn.unlift_program(mlp_apply)
    program_state = mlp_init_pure(
        num_layers=3,
        in_features=4,
        hidden_features=8,
        out_features=2,
    )
    output = mlp_apply_pure(program_state, x)
    print("Pure output:", output)

    state_def, params, static = nn.filter_state(
        program_state, lambda x: isinstance(x, jax.Array)
    )

    def loss_fn(params, x):
        state = nn.merge_state(state_def, params, static)
        preds = mlp_apply_pure(state, x)
        target = jnp.arange(2)
        return jnp.mean((preds - target) ** 2)

    @jax.jit
    def step(opt_state, params, x):
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        opt_state, params = optim.step(opt_state, params, grads)
        return opt_state, params, loss

    with optim.adamw():
        opt_state = optim.init(params)
        for _ in range(10):
            opt_state, params, loss = step(opt_state, params, x)
            print("Loss:", loss)


if __name__ == "__main__":
    main()
