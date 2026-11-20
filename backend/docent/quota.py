"""Daily question cap, counted in DynamoDB with one atomic conditional update per request."""
from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

DEFAULT_LIMIT = 150
TTL_DAYS = 3
KEY = "day"
COUNT = "count"
EXPIRES = "expires_at"  # the table's TTL attribute


def daily_limit() -> int:
    """`DAILY_LIMIT` from the environment; the default when unset or not a number; never below 0."""
    try:
        return max(0, int(os.environ.get("DAILY_LIMIT", DEFAULT_LIMIT)))
    except ValueError:
        return DEFAULT_LIMIT


def utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).date().isoformat()


def _expires_at(day: str) -> int:
    start = datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=UTC)
    return int((start + timedelta(days=TTL_DAYS)).timestamp())


def check_and_count(table, day: str, limit: int) -> bool:
    """Count one question for `day` and return True, or return False when `day` already has `limit`.

    One `UpdateItem` whose condition and increment DynamoDB applies atomically, so concurrent
    requests can never push the count past `limit`. `table` is a boto3 DynamoDB `Table`.
    Errors other than the failed condition propagate: the caller must fail closed.
    """
    if limit <= 0:
        return False
    try:
        table.update_item(
            Key={KEY: day},
            UpdateExpression="ADD #count :one SET #expires = if_not_exists(#expires, :expires)",
            ConditionExpression="attribute_not_exists(#count) OR #count < :limit",
            ExpressionAttributeNames={"#count": COUNT, "#expires": EXPIRES},
            ExpressionAttributeValues={":one": 1, ":limit": limit, ":expires": _expires_at(day)},
        )
    except Exception as exc:  # botocore ClientError; matched by code so tests need no boto3
        if getattr(exc, "response", {}).get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise
    return True
