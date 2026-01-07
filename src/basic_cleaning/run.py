#!/usr/bin/env python
"""
Este paso realiza una limpieza básica de los datos y guarda los resultados en W&B
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f'Download input artifact {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # drop outliers
    logger.info(f'Drop outliers thresholds: min {args.min_price}, max {args.max_price}')
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df_clean = df[idx].copy()

    # convert 'last_review' to datetime
    logger.info('Convert feature "last_review" to datetime type')
    df_clean['last_review'] = pd.to_datetime(df_clean['last_review'])

    # save cleaned dataframe
    logger.info(f'Save clean dataframe to {args.output_artifact}')
    df_clean.to_csv(args.output_artifact, index=False)

    # log artifact to Weights & Biases
    logger.info(f'W&B logging artifact {args.output_artifact}') 
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Este paso limpia los datos")

    parser.add_argument(
        '--input_artifact', 
        type=str,
        help='Input artifact',
        required=True
    )

    parser.add_argument(
        '--output_artifact', 
        type=str,
        help='Output file name',
        required=True
    )

    parser.add_argument(
        '--output_type', 
        type=str,
        help='Type of the output file',
        required=True
    )

    parser.add_argument(
        '--output_description', 
        type=str,
        help='Cleaned data',
        required=True
    )

    parser.add_argument(
        '--min_price', 
        type=float,
        help='Minimum price to filter the data',
        required=True
    )

    parser.add_argument(
        '--max_price', 
        type=float,
        help='Maximum price to filter the data',
        required=True
    )

    args = parser.parse_args()

    go(args)
