from scripts.check_device_smoke import check


def test_cpu_device_smoke() -> None:
    result = check("cpu")
    assert result["status"] == "PASS"
    assert result["maximum_repeat_error"] <= 1e-6
