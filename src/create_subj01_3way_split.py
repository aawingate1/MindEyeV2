#!/usr/bin/env python

import argparse
import os

from wds_3way_utils import materialize_subj01_three_way_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create the subj01 3-way WDS split layout.")
    parser.add_argument(
        "--source_wds_root",
        type=str,
        default="/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds",
        help="Existing WDS root containing subj01/train and subj01/new_test",
    )
    parser.add_argument(
        "--output_wds_root",
        type=str,
        default="/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds_3way",
        help="Destination root for the symlinked 3-way split layout",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = materialize_subj01_three_way_split(
        source_wds_root=args.source_wds_root,
        output_wds_root=args.output_wds_root,
    )
    print(f"Wrote 3-way split manifest to {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()
