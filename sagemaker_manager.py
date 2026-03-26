#!/usr/bin/env python3
#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Manage SageMaker endpoints for Deepgram STT and TTS services.

Spin up and tear down SageMaker endpoints used by the pipecat
DeepgramSageMakerSTTService and DeepgramSageMakerTTSService.

Deepgram models are deployed via AWS Marketplace model packages. You must
subscribe to the relevant Deepgram listings on AWS Marketplace first:

- STT: https://aws.amazon.com/marketplace/pp/prodview-pnd4oy3p2xtro
- TTS: https://aws.amazon.com/marketplace/pp/prodview-o2f7xiwm7khdk

After subscribing, find the model package ARN in the SageMaker console under
"Model packages > AWS Marketplace subscriptions" and set it in your .env.

Usage::

    # Create endpoints
    python sagemaker_manage.py up

    # Create only STT endpoint
    python sagemaker_manage.py up --service stt

    # Tear down endpoints
    python sagemaker_manage.py down

    # Tear down only TTS endpoint
    python sagemaker_manage.py down --service tts
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STT_ENDPOINT_NAME = "deepgram-stt"
DEFAULT_TTS_ENDPOINT_NAME = "deepgram-tts"
DEFAULT_STT_INSTANCE_TYPE = "ml.g5.2xlarge"
DEFAULT_TTS_INSTANCE_TYPE = "ml.g5.2xlarge"
DEFAULT_INSTANCE_COUNT = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sagemaker_client():
    """Build a boto3 SageMaker client from env vars.

    Reads AWS_SAGEMAKER_* env vars first, falling back to the
    corresponding AWS_* vars if the SageMaker-specific ones are not set.
    """
    region = os.getenv("AWS_SAGEMAKER_REGION") or os.getenv("AWS_REGION")
    if not region:
        logger.error(
            "AWS_SAGEMAKER_REGION (or AWS_REGION) is not set. "
            "Please configure it in your .env file."
        )
        sys.exit(1)

    kwargs: dict = {"region_name": region}

    access_key = os.getenv("AWS_SAGEMAKER_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = (
        os.getenv("AWS_SAGEMAKER_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    session_token = os.getenv("AWS_SAGEMAKER_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    return boto3.client("sagemaker", **kwargs)


def _get_iam_client():
    """Build a boto3 IAM client using the same credential resolution as SageMaker."""
    kwargs: dict = {}

    access_key = os.getenv("AWS_SAGEMAKER_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = (
        os.getenv("AWS_SAGEMAKER_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    session_token = os.getenv("AWS_SAGEMAKER_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    return boto3.client("iam", **kwargs)


SAGEMAKER_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

AUTO_CREATED_ROLE_NAME = "SageMakerExecutionRole"


def _find_sagemaker_execution_role() -> str | None:
    """Search the account for an existing IAM role with a SageMaker trust policy."""
    iam = _get_iam_client()
    logger.info("Searching for an existing IAM role with SageMaker trust policy ...")

    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page["Roles"]:
            doc = role.get("AssumeRolePolicyDocument", {})
            for stmt in doc.get("Statement", []):
                principal = stmt.get("Principal", {})
                service = principal.get("Service", "")
                services = [service] if isinstance(service, str) else service
                if "sagemaker.amazonaws.com" in services and stmt.get("Effect") == "Allow":
                    arn = role["Arn"]
                    logger.info(f"Found existing SageMaker role: {arn}")
                    return arn

    return None


def _create_sagemaker_execution_role() -> str:
    """Create a new IAM role for SageMaker with AmazonSageMakerFullAccess."""
    iam = _get_iam_client()
    logger.info(f"Creating IAM role '{AUTO_CREATED_ROLE_NAME}' ...")

    try:
        resp = iam.create_role(
            RoleName=AUTO_CREATED_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(SAGEMAKER_TRUST_POLICY),
            Description="SageMaker execution role (auto-created by sagemaker_manage.py)",
        )
        arn = resp["Role"]["Arn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "EntityAlreadyExists":
            arn = iam.get_role(RoleName=AUTO_CREATED_ROLE_NAME)["Role"]["Arn"]
            logger.info(f"Role '{AUTO_CREATED_ROLE_NAME}' already exists: {arn}")
        else:
            raise

    iam.attach_role_policy(
        RoleName=AUTO_CREATED_ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )
    logger.info(f"Attached AmazonSageMakerFullAccess to '{AUTO_CREATED_ROLE_NAME}'.")
    logger.info(f"Created SageMaker execution role: {arn}")
    return arn


def _resolve_execution_role() -> str:
    """Resolve the SageMaker execution role ARN.

    Priority:
    1. SAGEMAKER_EXECUTION_ROLE_ARN env var
    2. Discover an existing role with a SageMaker trust policy
    3. Create a new role automatically
    """
    from_env = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
    if from_env:
        logger.info(f"Using execution role from SAGEMAKER_EXECUTION_ROLE_ARN: {from_env}")
        return from_env

    logger.info("SAGEMAKER_EXECUTION_ROLE_ARN not set, looking for an existing role ...")
    discovered = _find_sagemaker_execution_role()
    if discovered:
        return discovered

    logger.info("No existing SageMaker role found, creating one ...")
    return _create_sagemaker_execution_role()


def _wait_for_endpoint(client, endpoint_name: str, poll_interval: int = 30):
    """Poll until the endpoint is InService or fails."""
    logger.info(f"Waiting for endpoint '{endpoint_name}' to become InService ...")
    while True:
        resp = client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        if status == "InService":
            logger.info(f"Endpoint '{endpoint_name}' is InService.")
            return
        if status in ("Failed", "RolledBack"):
            reason = resp.get("FailureReason", "unknown")
            logger.error(
                f"Endpoint '{endpoint_name}' entered status '{status}': {reason}"
            )
            sys.exit(1)
        logger.info(f"  Status: {status} — checking again in {poll_interval}s ...")
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Spin-up
# ---------------------------------------------------------------------------


def _create_endpoint(
    client,
    *,
    endpoint_name: str,
    model_package_arn: str,
    execution_role_arn: str,
    instance_type: str,
    instance_count: int,
    wait: bool,
):
    """Create a SageMaker Model, EndpointConfig, and Endpoint."""
    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # 1. Create model -------------------------------------------------------
    logger.info(f"Creating model '{model_name}' ...")
    try:
        client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=execution_role_arn,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            EnableNetworkIsolation=True,
        )
        logger.info(f"  Model '{model_name}' created.")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ValidationException" and "Cannot create already existing" in str(exc):
            logger.warning(f"  Model '{model_name}' already exists, reusing.")
        else:
            raise

    # 2. Create endpoint config ---------------------------------------------
    logger.info(f"Creating endpoint config '{config_name}' ...")
    try:
        client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": instance_count,
                    "InstanceType": instance_type,
                },
            ],
        )
        logger.info(f"  Endpoint config '{config_name}' created.")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ValidationException" and "Cannot create already existing" in str(exc):
            logger.warning(f"  Endpoint config '{config_name}' already exists, reusing.")
        else:
            raise

    # 3. Create endpoint ----------------------------------------------------
    logger.info(f"Creating endpoint '{endpoint_name}' ...")
    try:
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        logger.info(f"  Endpoint '{endpoint_name}' creation initiated.")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ValidationException" and "Cannot create already existing" in str(exc):
            logger.warning(f"  Endpoint '{endpoint_name}' already exists.")
            return
        else:
            raise

    if wait:
        _wait_for_endpoint(client, endpoint_name)


def _create_stt_endpoint(client, args, role_arn):
    """Create the STT endpoint."""
    pkg_arn = os.getenv("DEEPGRAM_STT_MODEL_PACKAGE_ARN")
    if not pkg_arn:
        logger.error(
            "DEEPGRAM_STT_MODEL_PACKAGE_ARN is not set. "
            "Subscribe to Deepgram STT on AWS Marketplace and set the model package ARN."
        )
        sys.exit(1)
    _create_endpoint(
        client,
        endpoint_name=args.stt_endpoint_name,
        model_package_arn=pkg_arn,
        execution_role_arn=role_arn,
        instance_type=args.stt_instance_type,
        instance_count=args.instance_count,
        wait=args.wait,
    )


def _create_tts_endpoint(client, args, role_arn):
    """Create the TTS endpoint."""
    pkg_arn = os.getenv("DEEPGRAM_TTS_MODEL_PACKAGE_ARN")
    if not pkg_arn:
        logger.error(
            "DEEPGRAM_TTS_MODEL_PACKAGE_ARN is not set. "
            "Subscribe to Deepgram TTS on AWS Marketplace and set the model package ARN."
        )
        sys.exit(1)
    _create_endpoint(
        client,
        endpoint_name=args.tts_endpoint_name,
        model_package_arn=pkg_arn,
        execution_role_arn=role_arn,
        instance_type=args.tts_instance_type,
        instance_count=args.instance_count,
        wait=args.wait,
    )


def cmd_up(args):
    """Handle the 'up' subcommand."""
    client = _get_sagemaker_client()
    role_arn = _resolve_execution_role()

    services = [args.service] if args.service else ["stt", "tts"]

    task_map = {"stt": _create_stt_endpoint, "tts": _create_tts_endpoint}
    tasks = {svc: task_map[svc] for svc in services}

    if len(tasks) > 1:
        logger.info("Creating STT and TTS endpoints in parallel ...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(fn, client, args, role_arn): svc for svc, fn in tasks.items()}
            for future in as_completed(futures):
                future.result()
    else:
        for fn in tasks.values():
            fn(client, args, role_arn)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Tear-down
# ---------------------------------------------------------------------------


def _delete_resource(client, delete_fn: str, name_kwarg: str, name: str):
    """Call a SageMaker delete API, ignoring 'not found' errors."""
    try:
        getattr(client, delete_fn)(**{name_kwarg: name})
        logger.info(f"  Deleted {delete_fn.replace('delete_', '').replace('_', ' ')} '{name}'.")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("ValidationException", "ResourceNotFound"):
            logger.warning(f"  {name} not found, skipping.")
        else:
            raise


def _delete_endpoint(client, endpoint_name: str):
    """Delete a SageMaker Endpoint, EndpointConfig, and Model."""
    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    logger.info(f"Tearing down endpoint '{endpoint_name}' ...")

    _delete_resource(client, "delete_endpoint", "EndpointName", endpoint_name)
    _delete_resource(client, "delete_endpoint_config", "EndpointConfigName", config_name)
    _delete_resource(client, "delete_model", "ModelName", model_name)


def cmd_down(args):
    """Handle the 'down' subcommand."""
    client = _get_sagemaker_client()

    services = [args.service] if args.service else ["stt", "tts"]

    endpoint_map = {"stt": args.stt_endpoint_name, "tts": args.tts_endpoint_name}
    endpoints = {svc: endpoint_map[svc] for svc in services}

    if len(endpoints) > 1:
        logger.info("Deleting STT and TTS endpoints in parallel ...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_delete_endpoint, client, name): svc
                for svc, name in endpoints.items()
            }
            for future in as_completed(futures):
                future.result()
    else:
        for name in endpoints.values():
            _delete_endpoint(client, name)

    logger.info("Done.")


def _get_endpoint_status(client, endpoint_name: str) -> dict | None:
    """Describe a SageMaker endpoint, returning None if it doesn't exist."""
    try:
        return client.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("ValidationException", "ResourceNotFound"):
            return None
        raise


def cmd_status(args):
    """Handle the 'status' subcommand (default when no command given)."""
    client = _get_sagemaker_client()

    endpoint_map = {"stt": args.stt_endpoint_name, "tts": args.tts_endpoint_name}

    found_any = False
    for svc, name in endpoint_map.items():
        info = _get_endpoint_status(client, name)
        if info is None:
            logger.info(f"  {svc.upper()} ({name}): not found")
        else:
            found_any = True
            status = info["EndpointStatus"]
            instance_count = ""
            try:
                config = client.describe_endpoint_config(
                    EndpointConfigName=f"{name}-config"
                )
                variant = config["ProductionVariants"][0]
                instance_count = (
                    f"  |  {variant['InitialInstanceCount']}x {variant['InstanceType']}"
                )
            except (ClientError, KeyError, IndexError):
                pass
            logger.info(f"  {svc.upper()} ({name}): {status}{instance_count}")
            if status in ("Failed", "RolledBack"):
                reason = info.get("FailureReason", "unknown")
                logger.info(f"    Reason: {reason}")

    if not found_any:
        logger.info("No active SageMaker endpoints found.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Manage SageMaker endpoints for Deepgram STT/TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables (set in .env):\n"
            "  AWS_SAGEMAKER_REGION             AWS region (falls back to AWS_REGION)\n"
            "  AWS_SAGEMAKER_ACCESS_KEY_ID      AWS credentials (falls back to AWS_ACCESS_KEY_ID)\n"
            "  AWS_SAGEMAKER_SECRET_ACCESS_KEY  AWS credentials (falls back to AWS_SECRET_ACCESS_KEY)\n"
            "  AWS_SAGEMAKER_SESSION_TOKEN      Session token (falls back to AWS_SESSION_TOKEN)\n"
            "  SAGEMAKER_EXECUTION_ROLE_ARN     IAM role ARN for SageMaker (auto-discovered/created if unset)\n"
            "  DEEPGRAM_STT_MODEL_PACKAGE_ARN   Marketplace model package ARN for Deepgram STT\n"
            "  DEEPGRAM_TTS_MODEL_PACKAGE_ARN   Marketplace model package ARN for Deepgram TTS\n"
        ),
    )

    # Shared endpoint-name flags
    for p in [parser]:
        p.add_argument(
            "--stt-endpoint-name",
            default=os.getenv("SAGEMAKER_STT_ENDPOINT_NAME", DEFAULT_STT_ENDPOINT_NAME),
            help=f"STT endpoint name (default: $SAGEMAKER_STT_ENDPOINT_NAME or '{DEFAULT_STT_ENDPOINT_NAME}')",
        )
        p.add_argument(
            "--tts-endpoint-name",
            default=os.getenv("SAGEMAKER_TTS_ENDPOINT_NAME", DEFAULT_TTS_ENDPOINT_NAME),
            help=f"TTS endpoint name (default: $SAGEMAKER_TTS_ENDPOINT_NAME or '{DEFAULT_TTS_ENDPOINT_NAME}')",
        )

    sub = parser.add_subparsers(dest="command")

    # -- status (also the default) ------------------------------------------
    status_parser = sub.add_parser("status", help="Show endpoint status (default)")
    status_parser.set_defaults(func=cmd_status)

    # -- up -----------------------------------------------------------------
    up_parser = sub.add_parser("up", help="Create SageMaker endpoints")
    up_parser.add_argument(
        "--service",
        choices=["stt", "tts"],
        default=None,
        help="Only create this service's endpoint (default: both)",
    )
    up_parser.add_argument(
        "--stt-instance-type",
        default=os.getenv("SAGEMAKER_STT_INSTANCE_TYPE", DEFAULT_STT_INSTANCE_TYPE),
        help=f"Instance type for STT (default: '{DEFAULT_STT_INSTANCE_TYPE}')",
    )
    up_parser.add_argument(
        "--tts-instance-type",
        default=os.getenv("SAGEMAKER_TTS_INSTANCE_TYPE", DEFAULT_TTS_INSTANCE_TYPE),
        help=f"Instance type for TTS (default: '{DEFAULT_TTS_INSTANCE_TYPE}')",
    )
    up_parser.add_argument(
        "--instance-count",
        type=int,
        default=DEFAULT_INSTANCE_COUNT,
        help=f"Number of instances per endpoint (default: {DEFAULT_INSTANCE_COUNT})",
    )
    up_parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for endpoints to become InService (default: true)",
    )
    up_parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Don't wait for endpoints — just initiate creation",
    )
    up_parser.set_defaults(func=cmd_up)

    # -- down ---------------------------------------------------------------
    down_parser = sub.add_parser("down", help="Delete SageMaker endpoints")
    down_parser.add_argument(
        "--service",
        choices=["stt", "tts"],
        default=None,
        help="Only delete this service's endpoint (default: both)",
    )
    down_parser.set_defaults(func=cmd_down)

    args = parser.parse_args()
    if args.command is None:
        args.func = cmd_status
    args.func(args)


if __name__ == "__main__":
    main()
