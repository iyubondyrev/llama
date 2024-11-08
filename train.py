import argparse
import random
import torch
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass
import wandb
from dataset import BinaryTokenDataset
from llama import LLAMA
from model_params import LLAMAParams
import numpy as np


def seed(base_seed: int = 42, /, *, one_cuda_seed: bool = False) -> int:
    """Set *diverse* global random seeds in `random`, `numpy` and `torch`.

    .. note::
        For all libraries, *different deterministically computed*
        (based on the ``base_seed`` argument) seeds are set to ensure that
        different libraries and (by default) devices generate diverse random numbers.

    **Usage**

    >>> import random
    >>> import numpy as np
    >>>
    >>> def f():
    ...     return (
    ...         random.randint(0, 10 ** 9),
    ...         np.random.rand(10).tolist(),
    ...         torch.randn(20).tolist(),
    ...     )
    ...
    >>> # Numbers sampled under the same random seed are equal.
    >>> delu.random.seed(0)
    >>> a = f()
    >>> delu.random.seed(0)
    >>> b = f()
    >>> a == b
    True

    Pass `None` to set a truly random seed generated by the OS:

    >>> # Save the generated `seed` for future reproducibility:
    >>> seed = delu.random.seed(None)
    >>> a = f()
    >>> # Reproduce the results:
    >>> delu.random.seed(seed)
    >>> b = f()
    >>> a == b
    True

    Args:
        base_seed: an integer from `[0, 2**64)` used to compute diverse seeds
            for all libraries. If `None`,
            then an unpredictable seed generated by OS is used and returned.
        one_cuda_seed: if `True`, then the same seed will be set for all CUDA devices,
            otherwise, different seeds will be set for all CUDA devices.
    Returns:
        the provided ``base_seed`` or the generated one if ``base_seed=None``.
    """
    # The implementation is based on:
    # - https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # - https://github.com/Lightning-AI/lightning/pull/6960#issuecomment-819672341

    sequence = np.random.SeedSequence(base_seed)
    _2_pow_64 = 2 ** 64

    def generate_state(*args, **kwargs) -> np.ndarray:
        new_sequence = sequence.spawn(1)[0]
        return new_sequence.generate_state(*args, **kwargs)

    # To generate a 128-bit seed for the standard library,
    # two uint64 numbers are generated and concatenated (literally).
    state_std = generate_state(2, dtype=np.uint64).tolist()
    random.seed(state_std[0] * _2_pow_64 + state_std[1])
    del state_std

    np.random.seed(generate_state(4))

    torch.manual_seed(int(generate_state(1, dtype=np.uint64)[0]))

    if not torch.cuda._is_in_bad_fork():
        if one_cuda_seed:
            torch.cuda.manual_seed_all(int(generate_state(1, dtype=np.uint64)[0]))
        else:
            if torch.cuda.is_available():
                torch.cuda.init()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.default_generators[i].manual_seed(
                        int(generate_state(1, dtype=np.uint64)[0])
                    )

    return base_seed

def main():
    parser = argparse.ArgumentParser(description="Train LLAMA model with specified parameters.")

    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--label_smoothing", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--num_epoch", type=int, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="chunks.txt.gz", help="Path to tokenized chunks dataset")

    args = parser.parse_args()

    seed(args.random_seed)

    dataset = BinaryTokenDataset(args.data_path)

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    params = LLAMAParams(
        dim=args.dim,
        vocab_size=32000, # tokenizer vocab size
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        num_epoch=args.num_epoch,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        random_seed=args.random_seed,
        logging_steps=args.logging_steps
    )
    model = LLAMA(params).to(params.device)
    model.assert_min_params(min_params=100 * 10**6)

    wandb.login()
    wandb.init(project="llama_project", name=args.wandb_run_name, config=vars(args))

    training_args = TrainingArguments(
        output_dir="./llama_model",
        overwrite_output_dir=True,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=700, 
        report_to="wandb",
        dataloader_num_workers=8,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()
