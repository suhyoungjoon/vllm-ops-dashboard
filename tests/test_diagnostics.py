from backend.diagnostics import diagnose
from backend.poller import VLLMMetrics


def _metrics(**overrides) -> VLLMMetrics:
    base = dict(
        kv_cache_usage_perc=0.3,
        num_requests_running=1.0,
        num_requests_waiting=0.0,
        prompt_tokens_total=10.0,
        generation_tokens_total=20.0,
        ttft_avg_seconds=0.1,
        request_success={"stop": 1.0},
        queue_time_avg_seconds=0.01,
        inference_time_avg_seconds=0.5,
        prefix_cache_hit_rate=0.8,
    )
    base.update(overrides)
    return VLLMMetrics(**base)


def test_diagnose_returns_ok_when_nothing_wrong():
    findings = diagnose(_metrics())
    assert len(findings) == 1
    assert findings[0].level == "ok"


def test_diagnose_flags_queue_bottleneck_when_queue_exceeds_inference_time():
    findings = diagnose(_metrics(queue_time_avg_seconds=1.0, inference_time_avg_seconds=0.2))
    levels = [f.level for f in findings]
    assert "warning" in levels
    assert any("대기열 병목" in f.message for f in findings)


def test_diagnose_ignores_tiny_queue_time_noise():
    findings = diagnose(_metrics(queue_time_avg_seconds=0.02, inference_time_avg_seconds=0.01))
    assert findings[0].level == "ok"


def test_diagnose_flags_kv_cache_pressure():
    findings = diagnose(_metrics(kv_cache_usage_perc=0.9))
    assert any(f.level == "warning" and "메모리 압박" in f.message for f in findings)


def test_diagnose_does_not_flag_kv_cache_below_threshold():
    findings = diagnose(_metrics(kv_cache_usage_perc=0.5))
    assert not any("메모리 압박" in f.message for f in findings)


def test_diagnose_flags_preemptions_when_delta_positive():
    findings = diagnose(_metrics(), preemptions_delta=2.0)
    assert any(f.level == "warning" and "선점 발생" in f.message and "2" in f.message for f in findings)


def test_diagnose_ignores_zero_preemptions_delta():
    findings = diagnose(_metrics(), preemptions_delta=0.0)
    assert not any("선점 발생" in f.message for f in findings)


def test_diagnose_flags_low_prefix_cache_hit_rate():
    findings = diagnose(_metrics(prefix_cache_hit_rate=0.2))
    assert any(f.level == "info" and "프리픽스 캐시 적중률" in f.message for f in findings)


def test_diagnose_ignores_prefix_cache_hit_rate_when_none():
    findings = diagnose(_metrics(prefix_cache_hit_rate=None))
    assert not any("프리픽스 캐시" in f.message for f in findings)


def test_diagnose_can_report_multiple_findings_at_once():
    findings = diagnose(
        _metrics(kv_cache_usage_perc=0.95, queue_time_avg_seconds=1.0, inference_time_avg_seconds=0.2),
        preemptions_delta=1.0,
    )
    assert len(findings) == 3
    assert all(f.level == "warning" for f in findings)
