import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

from docent import quota


class AWSError(Exception):
    code = ""

    def __init__(self):
        super().__init__(self.code)
        self.response = {"Error": {"Code": self.code}}


class ConditionFailed(AWSError):
    code = "ConditionalCheckFailedException"


class Throttled(AWSError):
    code = "ProvisionedThroughputExceededException"


class FakeTable:
    """Applies the one UpdateItem that `check_and_count` sends, atomically, like DynamoDB does."""

    def __init__(self):
        self.items: dict = {}
        self.calls: list = []
        self.lock = threading.Lock()

    def get_item(self, **kw):
        raise AssertionError("check_and_count must not read before it writes")

    def update_item(self, **kw):
        self.calls.append(kw)
        assert kw["UpdateExpression"] == "ADD #count :one SET #expires = if_not_exists(#expires, :expires)"
        assert kw["ConditionExpression"] == "attribute_not_exists(#count) OR #count < :limit"
        names, values = kw["ExpressionAttributeNames"], kw["ExpressionAttributeValues"]
        count_attr, expires_attr = names["#count"], names["#expires"]
        with self.lock:
            item = self.items.setdefault(kw["Key"]["day"], {"day": kw["Key"]["day"]})
            if count_attr in item and not item[count_attr] < values[":limit"]:
                raise ConditionFailed()
            item[count_attr] = item.get(count_attr, 0) + values[":one"]
            item.setdefault(expires_attr, values[":expires"])


def test_counts_up_to_the_limit_then_refuses():
    table = FakeTable()
    results = [quota.check_and_count(table, "2026-11-20", 3) for _ in range(5)]
    assert results == [True, True, True, False, False]
    assert table.items["2026-11-20"]["count"] == 3


def test_each_day_has_its_own_count():
    table = FakeTable()
    assert quota.check_and_count(table, "2026-11-20", 1)
    assert not quota.check_and_count(table, "2026-11-20", 1)
    assert quota.check_and_count(table, "2026-11-21", 1)


def test_ttl_is_three_days_after_the_day_starts():
    table = FakeTable()
    quota.check_and_count(table, "2026-11-20", 5)
    expected = int(datetime(2026, 11, 23, tzinfo=UTC).timestamp())
    assert table.items["2026-11-20"]["expires_at"] == expected


def test_zero_limit_refuses_without_a_write():
    table = FakeTable()
    assert quota.check_and_count(table, "2026-11-20", 0) is False
    assert table.calls == []


def test_concurrent_requests_never_pass_the_limit():
    table = FakeTable()
    results: list = []
    barrier = threading.Barrier(40)

    def worker():
        barrier.wait()
        results.append(quota.check_and_count(table, "2026-11-20", 25))

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 25
    assert table.items["2026-11-20"]["count"] == 25


def test_other_errors_propagate_so_the_caller_fails_closed():
    class Table(FakeTable):
        def update_item(self, **kw):
            raise Throttled()

    with pytest.raises(Throttled):
        quota.check_and_count(Table(), "2026-11-20", 5)


@pytest.mark.parametrize("value, expected", [(None, 150), ("10", 10), ("0", 0), ("-4", 0), ("lots", 150)])
def test_daily_limit_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DAILY_LIMIT", raising=False)
    else:
        monkeypatch.setenv("DAILY_LIMIT", value)
    assert quota.daily_limit() == expected


def test_utc_day_uses_utc():
    evening_west = datetime(2026, 11, 20, 20, 0, tzinfo=timezone(timedelta(hours=-8)))
    assert quota.utc_day(evening_west) == "2026-11-21"
