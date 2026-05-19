#!/usr/bin/env python

from models_mrae_retrieval_first import add_model_specific_args, build_model
from train_mrae_variant_common import run_training


if __name__ == "__main__":
    run_training("MRAE retrieval-first MindEye training", build_model, add_model_specific_args)
