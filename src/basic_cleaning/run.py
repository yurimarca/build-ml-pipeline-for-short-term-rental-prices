#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import tempfile

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with tempfile.NamedTemporaryFile(mode='wb+', suffix=".csv") as fp:

        logger.info("Basic cleaning step")

        run = wandb.init(job_type="basic_cleaning")
        run.config.update(args)

        # Download input artifact. This will also log that this script is using this
        # particular version of the artifact
        # artifact_local_path = run.use_artifact(args.input_artifact).file()

        logger.info("Downloading artifact: %s", args.input_artifact)
        artifact = run.use_artifact(args.input_artifact)
        artifact_path = artifact.file()

        df = pd.read_csv(artifact_path)

        logger.info("Cleaning artifact: %s", args.input_artifact)    
        # Duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        # Price range
        idx = df['price'].between(args.min_price, args.max_price)
        df = df[idx].copy()

        # Save dataframe to temp file
        df.to_csv(fp.name, header=True, index=False)

        # Log artifact to W&B
        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(fp.name)

        logger.info("Logging artifact: %s", args.output_artifact)
        run.log_artifact(artifact)

        if df.isna().sum().sum() > 0:
            nans = df.isna().sum().sum()
            logger.warning(f"WARNING: There are {nans} NaN cells in the cleaned data.")

        run.finish()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input raw dataset",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output clean dataset",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum priced allowed for the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum priced allowed for the dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)
