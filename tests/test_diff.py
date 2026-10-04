import json

import pytest
from fastapi.testclient import TestClient

from app.engine import ValidationError, diff
from app.main import app
from app.models import DiffRequest
from pydantic import ValidationError as PydanticValidationError


def call_diff(payload):
    return diff(DiffRequest.model_validate(payload))


# A source with one sensitive column (email), a filter and a projection
# export. Graphs are assembled per test to model baseline/candidate edits.
def graph(filter_refs, export_specs, source_id="users", export_id="export_users"):
    return {
        "nodes": [
            {
                "id": source_id,
                "type": "source",
                "columns": [
                    {"name": "uid", "sensitive": False},
                    {"name": "email", "sensitive": True},
                    {"name": "country", "sensitive": False},
                ],
            },
            {
                "id": "only_cn",
                "type": "filter",
                "input": source_id,
                "references": filter_refs,
            },
            {
                "id": export_id,
                "type": "projection",
                "input": "only_cn",
                "columns": export_specs,
            },
        ],
        "export": export_id,
    }


UID = {"name": "user_id", "copy": "uid"}
MAIL = {"name": "mail", "copy": "email"}
BUCKET = {"name": "bucket", "constant": "ALL"}
TAG = {"name": "tag", "constant": "v2"}

EMAIL = {"node_id": "users", "column": "email"}


def pair(baseline, candidate):
    return {"baseline": baseline, "candidate": candidate}


def by_name(res):
    return {col["name"]: col for col in res["columns"]}


# ----------------------------- happy paths --------------------------------


def test_identical_graphs_have_no_deltas_and_no_regression():
    g = graph(["country"], [UID, MAIL, BUCKET])
    res = call_diff(pair(g, g))
    assert res["regression"] is False
    assert [col["name"] for col in res["columns"]] == [
        "user_id",
        "mail",
        "bucket",
    ]
    for col in res["columns"]:
        assert col["content_reasons"] == {"added": [], "removed": []}
        assert col["row_reasons"] == {"added": [], "removed": []}
        assert col["regression"] is False
    assert res["baseline"]["blocked"] is True
    assert res["candidate"]["blocked"] is True


def test_content_reasons_added_and_removed_on_a_shared_column():
    # Baseline wipes mail with a constant; candidate copies the sensitive
    # column instead: a content reason appears.
    baseline = graph(["country"], [UID, {"name": "mail", "constant": "x"}])
    candidate = graph(["country"], [UID, MAIL])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    assert mail["baseline"]["blocked"] is False
    assert mail["candidate"]["blocked"] is True
    assert mail["content_reasons"] == {"added": [EMAIL], "removed": []}
    assert mail["regression"] is True
    assert res["regression"] is True

    # The reverse edit clears the content: a removed reason, no rollback.
    reverse = call_diff(pair(candidate, baseline))
    mail = by_name(reverse)["mail"]
    assert mail["content_reasons"] == {"added": [], "removed": [EMAIL]}
    assert mail["baseline"]["blocked"] is True
    assert mail["candidate"]["blocked"] is False
    assert mail["regression"] is False
    assert reverse["regression"] is False


def test_added_safe_column_is_not_a_regression():
    baseline = graph(["country"], [UID, BUCKET])
    candidate = graph(["country"], [UID, TAG, BUCKET])
    res = call_diff(pair(baseline, candidate))
    tag = by_name(res)["tag"]
    assert tag["baseline"] == {"blocked": None}
    assert tag["candidate"] == {"blocked": False}
    assert tag["content_reasons"] == {"added": [], "removed": []}
    assert tag["regression"] is False
    assert res["regression"] is False


def test_added_blocked_column_is_a_regression():
    baseline = graph(["country"], [UID, BUCKET])
    candidate = graph(["country"], [UID, MAIL, BUCKET])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    assert mail["baseline"] == {"blocked": None}
    assert mail["candidate"] == {"blocked": True}
    assert mail["content_reasons"] == {"added": [EMAIL], "removed": []}
    assert mail["regression"] is True
    assert res["regression"] is True


def test_deleted_blocked_column_is_not_a_regression():
    baseline = graph(["country"], [UID, MAIL, BUCKET])
    candidate = graph(["country"], [UID, BUCKET])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    assert mail["baseline"] == {"blocked": True}
    assert mail["candidate"] == {"blocked": None}
    # The absent side counts as the empty reason set.
    assert mail["content_reasons"] == {"added": [], "removed": [EMAIL]}
    assert mail["regression"] is False
    assert res["regression"] is False


def test_deleted_safe_column_is_not_a_regression():
    baseline = graph(["country"], [UID, MAIL, BUCKET])
    candidate = graph(["country"], [UID, MAIL])
    res = call_diff(pair(baseline, candidate))
    bucket = by_name(res)["bucket"]
    assert bucket["baseline"] == {"blocked": False}
    assert bucket["candidate"] == {"blocked": None}
    assert bucket["regression"] is False
    assert res["regression"] is False


def test_content_origin_turning_into_row_influence_is_a_regression():
    # Baseline filters on a non-sensitive column: mail carries sensitive
    # content only. Candidate filters on the sensitive email column: the same
    # origin now also shapes the surviving rows. The column is blocked on
    # both sides, but the new row reason is still a rollback.
    baseline = graph(["country"], [UID, MAIL])
    candidate = graph(["email"], [UID, MAIL])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    assert mail["baseline"]["blocked"] is True
    assert mail["candidate"]["blocked"] is True
    assert mail["content_reasons"] == {"added": [], "removed": []}
    assert mail["row_reasons"] == {"added": [EMAIL], "removed": []}
    assert mail["regression"] is True
    assert res["regression"] is True


def test_row_influence_turning_back_into_content_only_is_not_a_regression():
    baseline = graph(["email"], [UID, MAIL])
    candidate = graph(["country"], [UID, MAIL])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    # mail keeps the content reason in both versions; only the row reason
    # disappears, so no reason category gains anything.
    assert mail["content_reasons"] == {"added": [], "removed": []}
    assert mail["row_reasons"] == {"added": [], "removed": [EMAIL]}
    assert mail["regression"] is False
    assert res["regression"] is False


def test_row_influence_gained_by_a_clean_column_blocks_it():
    # user_id is a clean copy on both sides, but the candidate's sensitive
    # predicate taints every surviving row.
    baseline = graph(["country"], [UID, MAIL, BUCKET])
    candidate = graph(["email"], [UID, MAIL, BUCKET])
    res = call_diff(pair(baseline, candidate))
    cols = by_name(res)
    assert cols["user_id"]["row_reasons"] == {"added": [EMAIL], "removed": []}
    assert cols["user_id"]["regression"] is True
    assert cols["bucket"]["regression"] is True
    # mail already carried the content reason; the newly gained row reason
    # rolls back as well.
    assert cols["mail"]["regression"] is True
    assert res["regression"] is True


def test_origin_moving_from_rows_back_to_content_only_is_handled_per_kind():
    # Baseline: sensitive predicate (row influence), mail copied. Candidate:
    # non-sensitive predicate but mail additionally wiped by a constant: the
    # content reason disappears while the row reason disappears too. Only
    # removals exist, so no rollback.
    baseline = graph(["email"], [UID, MAIL, BUCKET])
    candidate = graph(["country"], [UID, {"name": "mail", "constant": "x"}, BUCKET])
    res = call_diff(pair(baseline, candidate))
    mail = by_name(res)["mail"]
    assert mail["content_reasons"] == {"added": [], "removed": [EMAIL]}
    assert mail["row_reasons"] == {"added": [], "removed": [EMAIL]}
    assert mail["regression"] is False
    assert res["regression"] is False


def test_column_order_shared_and_added_follow_candidate_then_deleted():
    baseline = graph(["country"], [UID, MAIL, BUCKET])
    # Candidate reorders, adds tag, drops bucket.
    candidate = graph(["country"], [TAG, MAIL, UID])
    res = call_diff(pair(baseline, candidate))
    assert [col["name"] for col in res["columns"]] == [
        "tag",
        "mail",
        "user_id",
        "bucket",
    ]


def test_deleted_columns_follow_baseline_order():
    baseline = graph(["country"], [UID, MAIL, BUCKET, TAG])
    candidate = graph(["country"], [UID])
    res = call_diff(pair(baseline, candidate))
    assert [col["name"] for col in res["columns"]] == [
        "user_id",
        "mail",
        "bucket",
        "tag",
    ]


def test_reasons_are_sorted_and_deduplicated_by_node_id_and_column():
    # Two sources feed a join on a sensitive key; the right branch is also
    # filtered on another sensitive column, contributing a second row origin
    # on that side. Baseline uses source s_b, candidate uses s_c: the added
    # and removed reason sets each span two columns of one node and must come
    # out sorted by (node_id, column).
    def build(first, second):
        return {
            "nodes": [
                {
                    "id": first,
                    "type": "source",
                    "columns": [
                        {"name": "pkey", "sensitive": True},
                        {"name": "v", "sensitive": False},
                    ],
                },
                {
                    "id": second,
                    "type": "source",
                    "columns": [
                        {"name": "pkey", "sensitive": True},
                        {"name": "pred", "sensitive": True},
                    ],
                },
                {
                    "id": "g",
                    "type": "filter",
                    "input": second,
                    "references": ["pred"],
                },
                {
                    "id": "j",
                    "type": "join",
                    "left_input": first,
                    "right_input": "g",
                    "left_key": "pkey",
                    "right_key": "pkey",
                    "outputs": [
                        {"alias": "v", "side": "left", "column": "v"},
                    ],
                },
            ],
            "export": "j",
        }

    baseline = build("s_a", "s_b")
    candidate = build("s_a", "s_c")
    res = call_diff(pair(baseline, candidate))
    col = res["columns"][0]
    assert col["row_reasons"]["added"] == [
        {"node_id": "s_c", "column": "pkey"},
        {"node_id": "s_c", "column": "pred"},
    ]
    assert col["row_reasons"]["removed"] == [
        {"node_id": "s_b", "column": "pkey"},
        {"node_id": "s_b", "column": "pred"},
    ]
    assert col["regression"] is True


def test_same_column_name_pairs_across_different_lineages():
    # The exported column keeps the name "mail" in both versions even though
    # the candidate renames its source lineage; pairing is by export name.
    renamed_baseline = {
        "nodes": [
            {
                "id": "users",
                "type": "source",
                "columns": [
                    {"name": "uid", "sensitive": False},
                    {"name": "email_addr", "sensitive": True},
                ],
            },
            {
                "id": "e",
                "type": "projection",
                "input": "users",
                "columns": [{"name": "mail", "copy": "email_addr"}],
            },
        ],
        "export": "e",
    }
    renamed_candidate = {
        "nodes": [
            {
                "id": "users",
                "type": "source",
                "columns": [
                    {"name": "uid", "sensitive": False},
                    {"name": "email_address", "sensitive": True},
                ],
            },
            {
                "id": "e",
                "type": "projection",
                "input": "users",
                "columns": [{"name": "mail", "copy": "email_address"}],
            },
        ],
        "export": "e",
    }
    res = call_diff(pair(renamed_baseline, renamed_candidate))
    assert [col["name"] for col in res["columns"]] == ["mail"]
    mail = res["columns"][0]
    # Different source column names on each side -> one removed, one added.
    assert mail["content_reasons"] == {
        "added": [{"node_id": "users", "column": "email_address"}],
        "removed": [{"node_id": "users", "column": "email_addr"}],
    }
    assert mail["regression"] is True


def test_reviews_embed_both_versions_full_results():
    baseline = graph(["country"], [UID, BUCKET])
    candidate = graph(["country"], [UID, TAG, BUCKET])
    res = call_diff(pair(baseline, candidate))
    assert res["baseline"]["export"] == "export_users"
    assert [c["name"] for c in res["baseline"]["columns"]] == [
        "user_id",
        "bucket",
    ]
    assert [c["name"] for c in res["candidate"]["columns"]] == [
        "user_id",
        "tag",
        "bucket",
    ]


# ----------------------------- whole-graph validation ---------------------


def expect_error(payload, code):
    with pytest.raises(ValidationError) as exc:
        call_diff(payload)
    assert exc.value.code == code


def test_invalid_baseline_rejects_the_whole_request():
    bad = graph(["country"], [UID, MAIL])
    bad["export"] = "ghost"
    good = graph(["country"], [UID, MAIL])
    expect_error(pair(bad, good), "INVALID_REFERENCE")


def test_invalid_candidate_rejects_the_whole_request():
    good = graph(["country"], [UID, MAIL])
    bad = graph(["country"], [UID, MAIL])
    bad["export"] = "ghost"
    expect_error(pair(good, bad), "INVALID_REFERENCE")


def test_baseline_checked_before_candidate():
    bad_base = graph(["country"], [UID])
    bad_base["export"] = "missing-base"
    bad_cand = graph(["country"], [UID])
    bad_cand["export"] = "missing-cand"
    expect_error(pair(bad_base, bad_cand), "INVALID_REFERENCE")
    with pytest.raises(ValidationError) as exc:
        call_diff(pair(bad_base, bad_cand))
    assert exc.value.node_id == "missing-base"


def test_baseline_cycle_rejected_like_review():
    good = graph(["country"], [UID])
    bad = {
        "nodes": [
            {"id": "a", "type": "projection", "input": "b",
             "columns": [{"name": "x", "constant": 1}]},
            {"id": "b", "type": "projection", "input": "a",
             "columns": [{"name": "x", "constant": 1}]},
        ],
        "export": "a",
    }
    expect_error(pair(bad, good), "CYCLE")


def test_invalid_graph_returns_no_partial_diff():
    good = graph(["country"], [UID])
    bad = graph(["country"], [UID, MAIL])
    del bad["nodes"][0]["columns"][1]
    # The projection still copies the now-missing "email" column.
    with pytest.raises(ValidationError) as exc:
        call_diff(pair(good, bad))
    assert exc.value.code == "INVALID_REFERENCE"
    assert not hasattr(exc.value, "columns")


# ----------------------------- input immutability -------------------------


def test_input_graphs_are_not_modified():
    baseline = graph(["country"], [UID, MAIL, BUCKET])
    candidate = graph(["email"], [UID, MAIL, TAG])
    snapshot_b = json.loads(json.dumps(baseline))
    snapshot_c = json.loads(json.dumps(candidate))
    call_diff(pair(baseline, candidate))
    assert baseline == snapshot_b
    assert candidate == snapshot_c


# ----------------------------- schema validation --------------------------


def test_schema_rejects_unknown_fields():
    payload = {
        "baseline": graph(["country"], [UID]),
        "candidate": graph(["country"], [UID]),
        "extra": 1,
    }
    with pytest.raises(PydanticValidationError):
        DiffRequest.model_validate(payload)


def test_schema_rejects_missing_version():
    with pytest.raises(PydanticValidationError):
        DiffRequest.model_validate({"baseline": graph(["country"], [UID])})


def test_schema_rejects_non_graph_version():
    with pytest.raises(PydanticValidationError):
        DiffRequest.model_validate(
            {"baseline": "nope", "candidate": graph(["country"], [UID])}
        )


def test_schema_rejects_more_than_60_nodes_in_either_graph():
    nodes = [{"id": "s0", "type": "source", "columns": [{"name": "a"}]}]
    for i in range(1, 61):
        nodes.append(
            {
                "id": f"n{i}",
                "type": "projection",
                "input": f"n{i - 1}" if i > 1 else "s0",
                "columns": [{"name": "a", "copy": "a"}],
            }
        )
    too_big = {"nodes": nodes, "export": "n60"}
    good = graph(["country"], [UID])
    with pytest.raises(PydanticValidationError):
        DiffRequest.model_validate({"baseline": too_big, "candidate": good})
    with pytest.raises(PydanticValidationError):
        DiffRequest.model_validate({"baseline": good, "candidate": too_big})


# ----------------------------- HTTP endpoint ------------------------------


client = TestClient(app)


def test_http_diff_success():
    with open("examples/diff_request.json") as fh:
        payload = json.load(fh)
    resp = client.post("/api/diff", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["regression"] is True
    by = {col["name"]: col for col in body["columns"]}
    # mail: the sensitive origin moves from content influence to row
    # influence while the column stays blocked on both sides.
    mail = by["mail"]
    assert mail["baseline"]["blocked"] is True
    assert mail["candidate"]["blocked"] is True
    assert mail["content_reasons"] == {"added": [], "removed": []}
    assert mail["row_reasons"] == {"added": [EMAIL], "removed": []}
    assert mail["regression"] is True
    # New column tag is row-blocked in the candidate.
    assert by["tag"]["baseline"]["blocked"] is None
    assert by["tag"]["regression"] is True
    # Deleted safe column bucket is not a rollback.
    assert by["bucket"]["candidate"]["blocked"] is None
    assert by["bucket"]["regression"] is False
    # Column order: candidate order first, deleted baseline columns last.
    assert [col["name"] for col in body["columns"]] == [
        "user_id",
        "mail",
        "tag",
        "bucket",
    ]


def test_http_diff_invalid_baseline_returns_400_with_old_error_format():
    with open("examples/diff_request.json") as fh:
        payload = json.load(fh)
    payload["baseline"]["export"] = "ghost"
    resp = client.post("/api/diff", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "INVALID_REFERENCE"
    assert body["error"]["node_id"] == "ghost"
    assert "columns" not in body and "baseline" not in body


def test_http_diff_invalid_candidate_returns_400():
    with open("examples/diff_request.json") as fh:
        payload = json.load(fh)
    payload["candidate"]["nodes"][0]["id"] = "people"
    resp = client.post("/api/diff", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REFERENCE"


def test_http_diff_schema_error_returns_422():
    resp = client.post(
        "/api/diff",
        json={"baseline": graph(["country"], [UID])},
    )
    assert resp.status_code == 422

    with open("examples/diff_request.json") as fh:
        payload = json.load(fh)
    payload["unknown"] = True
    assert client.post("/api/diff", json=payload).status_code == 422


def test_http_other_endpoints_unchanged():
    good = graph(["country"], [UID, MAIL, BUCKET])
    resp = client.post("/api/review", json=good)
    assert resp.status_code == 200
    assert resp.json()["blocked"] is True
    assert "regression" not in resp.json()
