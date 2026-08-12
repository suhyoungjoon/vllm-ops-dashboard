from backend.capacity import CapacityEstimate
from backend.diagnostics import diagnose
from backend.poller import VLLMMetrics


def _capacity(**overrides) -> CapacityEstimate:
    base = dict(
        max_concurrency=40.0,
        current_concurrency=5.0,
        headroom=35.0,
        headroom_ratio=0.125,
        kv_trend_per_minute=None,
        minutes_to_saturation=None,
    )
    base.update(overrides)
    return CapacityEstimate(**base)


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
    assert findings[0].recommendation is None


def test_diagnose_flags_queue_bottleneck_when_queue_exceeds_inference_time():
    findings = diagnose(_metrics(queue_time_avg_seconds=1.0, inference_time_avg_seconds=0.2))
    levels = [f.level for f in findings]
    assert "warning" in levels
    match = next(f for f in findings if "대기열 병목" in f.message)
    assert match.recommendation is not None
    assert "max_num_seqs" in match.recommendation


def test_diagnose_ignores_tiny_queue_time_noise():
    findings = diagnose(_metrics(queue_time_avg_seconds=0.02, inference_time_avg_seconds=0.01))
    assert findings[0].level == "ok"


def test_diagnose_flags_kv_cache_pressure():
    findings = diagnose(_metrics(kv_cache_usage_perc=0.9))
    match = next(f for f in findings if "메모리 압박" in f.message)
    assert match.level == "warning"
    assert "gpu_memory_utilization" in match.recommendation


def test_diagnose_does_not_flag_kv_cache_below_threshold():
    findings = diagnose(_metrics(kv_cache_usage_perc=0.5))
    assert not any("메모리 압박" in f.message for f in findings)


def test_diagnose_flags_preemptions_when_delta_positive():
    findings = diagnose(_metrics(), preemptions_delta=2.0)
    match = next(f for f in findings if "선점 발생" in f.message)
    assert match.level == "warning" and "2" in match.message
    assert "max_model_len" in match.recommendation


def test_diagnose_ignores_zero_preemptions_delta():
    findings = diagnose(_metrics(), preemptions_delta=0.0)
    assert not any("선점 발생" in f.message for f in findings)


def test_diagnose_flags_low_prefix_cache_hit_rate():
    findings = diagnose(_metrics(prefix_cache_hit_rate=0.2))
    match = next(f for f in findings if "프리픽스 캐시 적중률" in f.message)
    assert match.level == "info"
    assert "템플릿화" in match.recommendation


def test_diagnose_ignores_prefix_cache_hit_rate_when_none():
    findings = diagnose(_metrics(prefix_cache_hit_rate=None))
    assert not any("프리픽스 캐시" in f.message for f in findings)


def test_diagnose_flags_saturation_imminent_within_threshold():
    findings = diagnose(_metrics(), capacity=_capacity(minutes_to_saturation=3.0))
    match = next(f for f in findings if "포화 임박" in f.message)
    assert match.level == "warning"
    assert "3.0" in match.message
    assert match.recommendation is not None


def test_diagnose_ignores_saturation_beyond_warning_window():
    findings = diagnose(_metrics(), capacity=_capacity(minutes_to_saturation=30.0))
    assert not any("포화 임박" in f.message for f in findings)


def test_diagnose_ignores_saturation_when_not_trending_up():
    findings = diagnose(_metrics(), capacity=_capacity(minutes_to_saturation=None))
    assert not any("포화 임박" in f.message for f in findings)


def test_diagnose_ignores_saturation_when_capacity_not_provided():
    findings = diagnose(_metrics())
    assert not any("포화 임박" in f.message for f in findings)


def test_diagnose_can_report_multiple_findings_at_once():
    findings = diagnose(
        _metrics(kv_cache_usage_perc=0.95, queue_time_avg_seconds=1.0, inference_time_avg_seconds=0.2),
        preemptions_delta=1.0,
    )
    assert len(findings) == 3
    assert all(f.level == "warning" for f in findings)
