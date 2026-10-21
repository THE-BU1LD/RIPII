"""Append the authorized radial study without rewriting predecessor decisions."""

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "research/successors/radial_adaptation_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = REPO.parent / "RESEARCH_STATE.json"
    original = path.read_text()
    portfolio = json.loads(original)
    before = copy.deepcopy(portfolio)
    project = next(p for p in portfolio["projects"] if p["id"] == "PR-025")
    prior = copy.deepcopy(project)
    state = json.loads((STUDY / "STATE.json").read_text())
    entry = {
        "study_id": state["study"],
        "state": "RIPII/research/successors/radial_adaptation_v1/STATE.json",
        "phase": state["status"],
        "scientific_disposition": state["scientific_result"],
        "authorization": "2026-10-21 user requested a properly trained new adaptive-versus-fixed radial study and selected free Kaggle only, 4 allocated GPU-hours and $0.",
        "budget": state["budget"],
        "predecessor_preserved": True,
        "test_data_opened": state["test_data_opened"],
        "scope": "Prospective synthetic successor; no real-world or publication claim.",
        "url": state["url"],
    }
    entries = project.setdefault("successors", [])
    found = [e for e in entries if e["study_id"] == entry["study_id"]]
    if len(found) > 1:
        raise ValueError("Duplicate successor identity")
    if found:
        found[0].clear()
        found[0].update(entry)
    else:
        entries.append(entry)
    engineering_path = REPO / "research/engineering/radial_execution_v1/STATE.json"
    if engineering_path.exists():
        engineering = json.loads(engineering_path.read_text())
        engineering_entry = {
            "study_id": engineering["study"],
            "state": "RIPII/research/engineering/radial_execution_v1/STATE.json",
            "phase": engineering["status"],
            "scientific_disposition": "NOT_EVALUATED",
            "scope": "Engineering execution equivalence only; no scientific fits or test access",
            "charge": engineering["charge"],
            "predecessor_preserved": True,
        }
        existing = [e for e in entries if e["study_id"] == engineering_entry["study_id"]]
        if len(existing) > 1:
            raise ValueError("Duplicate engineering identity")
        if existing:
            existing[0].clear()
            existing[0].update(engineering_entry)
        else:
            entries.append(engineering_entry)
        source = "research/engineering/radial_execution_v1/STATE.json"
        if source not in project["status_sources"]:
            project["status_sources"].append(source)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    project["history"].append(
        f"{now} New radial-adaptation v1 indexed: {state['status']}; "
        f"scientific result {state['scientific_result']}; test data opened "
        f"{state['test_data_opened']}. Four allocated GPU-hours/$0; "
        "all predecessor dispositions preserved. See successor STATE and retained receipts."
    )
    project["next_gate"] = state.get(
        "next_action",
        "Read the radial-adaptation v1 retained execution receipt before any further run.",
    )
    source = "research/successors/radial_adaptation_v1/STATE.json"
    if source not in project["status_sources"]:
        project["status_sources"].append(source)
    for field in ("decisions", "disposition", "terminal", "claims"):
        assert project[field] == prior[field], field
    for old, new in zip(before["projects"], portfolio["projects"], strict=True):
        if old["id"] != "PR-025":
            assert old == new

    # Replace only PR-025's original serialized block, preserving unrelated bytes.
    def block(value):
        return "\n".join(
            "    " + line
            for line in json.dumps(value, indent=2, ensure_ascii=False).splitlines()
        )

    old, new = block(prior), block(project)
    if original.count(old) != 1:
        raise RuntimeError(
            "Portfolio formatting changed; narrow replacement unavailable"
        )
    updated = original.replace(old, new, 1)
    assert json.loads(updated) == portfolio
    receipt = {
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "project": "PR-025",
        "phase": entry["phase"],
        "applied": args.apply,
    }
    if args.apply:
        if path.read_text() != original:
            raise RuntimeError("Concurrent portfolio modification")
        path.write_text(updated)
        assert json.loads(path.read_text()) == portfolio
        (STUDY / "PORTFOLIO_RECONCILIATION.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
