#!/usr/bin/env python

from models_mrae_latent_transformer_retrieval import add_model_specific_args, build_model
from train_mrae_variant_common import run_training


if __name__ == "__main__":
    run_training(
        "MRAE latent transformer with retrieval-first auxiliary losses",
        build_model,
        add_model_specific_args,
    )
