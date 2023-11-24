# %%
import array
import functools as ft
import gzip
import os
import struct
import urllib.request

import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Tuple
import equinox as eqx
from tqdm import trange

from mixermlp import Mixer2d

from icecream import ic 


def mnist():
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)


def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size


def single_loss_fn_sum(model, t, sample, int_beta, weight,  eps=1e-5,  * ,key):
    mean = sample * jnp.exp(-0.5 * int_beta(t))
    var =  jnp.clip(1-jnp.exp(-int_beta(t)), a_min=eps)
    std = jnp.sqrt(var)
    noise = jr.normal(key, mean.shape)
    p_0t = mean + std * noise
    pred_score = model(p_0t, t)
    return weight(t) * jnp.sum((pred_score + noise/std) ** 2)


def single_loss_fn(model, t, sample, int_beta, weight,  eps=1e-5,  * ,key):
    mean = sample * jnp.exp(-0.5 * int_beta(t))
    var =  jnp.maximum(1-jnp.exp(-int_beta(t)), eps)#, a_min=eps)
    std = jnp.sqrt(var)
    noise = jr.normal(key, mean.shape)
    p_0t = mean + std * noise
    # pred_score = model(p_0t, t)
    pred_score = model(t, p_0t)
    return weight(t) * jnp.mean((pred_score + noise/std) ** 2)


def batch_loss_fn(model, batch, int_beta,  weight, t_max, *, key):
    time_key, noise_key = jr.split(key, 2)

    # sample time from uniform distribution in [0, t_max]
    # t = jr.uniform(time_key, (batch.shape[0],)) * t_max
    
    # alternative sampling scheme
    batch_size = batch.shape[0]
    t = jr.uniform(time_key, (batch_size,), maxval=t_max/batch_size)
    t += jnp.arange(batch_size) * t_max/batch_size

    # average loss over uniform time samples
    noise_key = jr.split(noise_key, batch.shape[0])
    loss_fn = ft.partial(single_loss_fn, 
                         model=model, 
                         int_beta=int_beta, 
                         weight=weight)
    loss_fn = jax.vmap(loss_fn)
    loss_val = loss_fn(t=t, sample=batch, key=noise_key) 
    return jnp.mean(loss_val)


@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt: optax.GradientTransformation, 
    opt_state: optax.OptState, 
    batch: Array,
    int_beta: Callable,
    weight: Callable,
    t_max: float,
    key: PRNGKeyArray
) -> Tuple[eqx.Module, optax.OptState, Array]:
    loss_fn_val_grad = eqx.filter_value_and_grad(batch_loss_fn)
    loss_val, grad = loss_fn_val_grad(
        model,
        batch=batch,
        int_beta=int_beta,
        weight=weight,  
        t_max=t_max,
        key=key
    )
    updates, opt_state = opt.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return model, opt_state, loss_val, key



@eqx.filter_jit
def single_sample_fn(
        model: eqx.Module, 
        int_beta: Callable,
        data_shape, 
        dt0: float, 
        t_max: float, 
        key: PRNGKeyArray):
    # basic form SDE: dx = f(x, t) dt + g(t)dw
    #       where f(x, t) = -0.5 beta(t) x(t)
    #             g(t) = sqrt(beta(t))
    # with this form, we can write the reverse as:
    #   dx = [f(x, t) + 0.5 g(t)^2 \nabla_x log p(x, t)] dt
    #      = [-0.5 beta(t) x(t) - 0.5 beta(t) s_\theta(x, t)] dt
    #      = [-0.5 beta(t) (x(t) + s_\theta(x, t))] dt
    # there is no noise  → just the drift term
    def drift(t, x, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (x + model(t, x))
    
    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t_0 = 0.
    x_t_max = jr.normal(key, data_shape)

    # solve from t_max to t_0
    sol = dfx.diffeqsolve(term, solver, t_max, t_0, -dt0, x_t_max)
    return sol.ys[0]


def main(
    # Model hyperparameters
    t_max: float = 10.,
    # Training hyperparameters
    num_steps: int = 100_000, #1_000_000,
    lr: float = 3e-4,
    batch_size: int = 256,
    print_every: int = 1_000,
    # Sampling hyperparameters
    dt0: float = 0.1,
    sample_size: int = 10,
    # Seed
    seed: int = 0,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, train_key, sample_key = jr.split(key, 4)
    data = mnist()
    data_mean = data.mean()
    data_std = data.std()
    data_min, data_max = data.min(), data.max()
    data = (data - data_mean) / data_std

    loader = dataloader(data, batch_size, key=data_key)

    ic(data.shape)

    model = Mixer2d(
        img_size=data.shape[1:],
        patch_size=4,
        hidden_size=64,
        mix_patch_size=512,
        mix_hidden_size=512,
        num_blocks=4,
        t1=t_max,
        key=model_key,
    )

    int_beta = lambda t: t
    weight_fn = lambda t: 1 - jnp.exp(-int_beta(t))

    opt = optax.adam(learning_rate=lr) 
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    for step, batch in zip(pbar := trange(num_steps, ncols=80), 
                           loader):
        model, opt_state,loss_val, train_key = make_step(
            model=model,
            opt=opt,
            opt_state=opt_state,
            batch=batch,
            int_beta=int_beta,
            weight=weight_fn,
            t_max=t_max,
            key=train_key,
        )
        pbar.set_postfix({"Loss": f"{loss_val: .5f}"})

        if step % print_every == 0:
        #     print(f"Step {step}: Loss = {loss_val}")
            sample_key, run_key = jr.split(sample_key)
            run_key = jr.split(run_key, sample_size**2)

            sample_fn = ft.partial(single_sample_fn, model, int_beta, data.shape[1:], dt0, t_max)
            sample = jax.vmap(sample_fn)(run_key)
            sample = data_mean + data_std * sample
            sample = jnp.clip(sample, data_min, data_max)
            sample = einops.rearrange(
                sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
            )
            plt.imshow(sample, cmap="Greys")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    sample_key, run_key = jr.split(sample_key)
    run_key = jr.split(run_key, sample_size**2)

    sample_fn = ft.partial(single_sample_fn, model, int_beta, data.shape[1:], dt0, t_max)
    sample = jax.vmap(sample_fn)(run_key)
    sample = data_mean + data_std * sample
    # sample = jnp.clip(sample, data_min, data_max)
    sample = einops.rearrange(
        sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    plt.imshow(sample, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

# %%
