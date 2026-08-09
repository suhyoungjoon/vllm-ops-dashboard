# 프로젝트: vLLM Ops Dashboard (가칭)

## 1. 프로젝트 목적

vLLM(오픈소스 LLM 추론 서빙 엔진)의 `/metrics` 엔드포인트를 스크래핑해서
**Prometheus/Grafana 설치 없이** 실시간으로 상태를 보여주는 초경량 대시보드를 만든다.

### 비즈니스 배경 (참고용, 코드와 무관)
- 최종 목표는 "vLLM 추론 서빙 최적화 컨설팅" 사업의 첫 제품/포트폴리오.
- 이 앱 자체가 스터디 목적 + 향후 오픈코어 제품의 씨앗(seed)이 될 수 있음.
- 차별점: 기존엔 Prometheus + Grafana를 따로 설치/설정해야 하는데,
  이 도구는 "vLLM 서버 URL 하나만 입력하면 바로 대시보드가 뜨는" 제로 컨피그를 지향.

## 2. 개발 환경 제약 (중요)

- 개발 환경: **macOS (Apple Silicon)**, NVIDIA GPU 없음.
- vLLM은 CUDA 기반이지만, 이 맥에는 `~/.venv-vllm-metal`에 **MLX 기반 실제 Metal GPU 백엔드**(`vllm_metal` 플러그인, `device: mps`)가 설치되어 있어 CPU 전용은 아님. 다만 8GB unified memory라 성능은 여전히 제한적.
- **따라서 실제 vLLM 서버를 로컬에서 상시 띄워놓고 개발하지 않는다.**
  대신 아래 순서로 진행:
  1. (1회성) 맥에 vLLM을 실험적으로 설치해서 `/metrics` 응답 원본 포맷을 실제로 한 번 확인하고 샘플로 저장해둔다.
  2. 이후 대시보드 개발은 **mock 메트릭 서버**(아래 5번 참고)를 vLLM 대역으로 사용해서 진행한다.
  3. 최종 검증만 실제 GPU 서버(클라우드 등)에 붙여서 확인한다.
- 따라서 앱은 **"vLLM 서버 주소"만 설정으로 받고, 응답 포맷(Prometheus text exposition format)만 맞으면 실제 vLLM이든 mock이든 동일하게 동작해야 한다.** (인터페이스를 이 가정에 맞게 느슨하게 설계할 것)

### 2026-08-09 실측 메모 (8GB 맥, Qwen2.5-0.5B-Instruct 기준)
- **병목은 vLLM/Apple Silicon 자체가 아니라 8GB unified memory 압박(swap)이었음.** Docker Desktop(~900MB)과 잊고 있던 예전 `vllm serve` 프로세스(4일 전부터 방치)가 메모리를 점유한 상태에서는 5토큰짜리 요청도 90초 내 응답을 못 받았음.
- 해당 프로세스들을 정리(Docker 종료, 방치된 vLLM 프로세스 종료)하니 Metal 가용 메모리가 1.7GB → 2.6~2.9GB로 늘었고, KV 캐시 예산도 2.04GB → 4.28GB로 2배 증가.
- 정리 후 실측: 5토큰 요청 10.9초, 100토큰 요청 15초(~7토큰/초), **동시 요청 3개(각 40토큰)도 15.3초에 모두 완료**.
- **결론**: 이 8GB 맥으로도 "포트폴리오 데모(1~3명, 짧은 응답)" 수준은 가능함. 단, 개발 세션(Claude Code 등)과 데모를 동시에 돌리면 다시 압박이 걸릴 수 있으니 데모 직전엔 불필요한 앱을 정리할 것. 맥미니 16GB는 이 여유를 더 확보해주지만 "필수"는 아님. 실제 고객 트래픽급 동시성/처리량 검증에는 여전히 클라우드 GPU가 필요함(이 결론은 불변).

## 3. MVP 스코프 (v0.1 — 딱 이것만 구현)

### 포함
- [ ] 설정 화면(or 시작 시 env/CLI 인자)에서 vLLM `/metrics` URL 입력
- [ ] 백엔드가 N초(기본 2초)마다 해당 URL을 폴링해서 Prometheus 텍스트 포맷 파싱
- [ ] 파싱한 값 중 아래 핵심 지표만 우선 추출 (2026-08-05, 실제 vLLM-metal 서버에서 `curl /metrics`로 검증된 이름):
  - `vllm:kv_cache_usage_perc` (KV 캐시 사용률, Gauge, 0~1) — ⚠️ `gpu_cache_usage_perc`가 아니라 `kv_cache_usage_perc`가 맞는 이름
  - `vllm:num_requests_running` (현재 실행 중 요청 수, Gauge)
  - `vllm:num_requests_waiting` (대기 중 요청 수, Gauge)
  - `vllm:prompt_tokens_total` (누적 입력 토큰, Counter)
  - `vllm:generation_tokens_total` (누적 출력 토큰, Counter → 초당 증가량으로 TPS 환산)
  - `vllm:time_to_first_token_seconds_sum` / `_count` (TTFT, Histogram — `_sum ÷ _count`로 평균 계산)
  - `vllm:request_success_total` (완료된 요청 수, `finished_reason` 라벨로 stop/length/abort/error 구분)

  **파싱 시 주의**: 모든 지표에 `engine="0"`, `model_name="..."` 두 개의 라벨이 함께 붙는다. Prometheus 텍스트 파서(`prometheus_client.parser.text_string_to_metric_families`)로 파싱한 뒤, 라벨 딕셔너리에서 이 값들을 확인해서 필터링할 것.

  참고용 실제 샘플 원본: `vllm_metrics_sample.txt` (프로젝트 루트에 저장해두고, mock 서버가 이 포맷을 그대로 흉내내도록 참고 자료로 사용)
- [ ] WebSocket으로 프론트에 실시간 push
- [ ] 프론트: 카드 4개(KV캐시%, 실행중, 대기중, TPS) + 라인차트 1~2개(최근 5~10분)
- [ ] KV 캐시 사용률 90% 초과 시 카드 색상 경고(빨강)로 전환

### 명시적으로 제외 (v0.2 이후, 지금 만들지 말 것)
- 알림/Slack 연동, 이메일
- 다중 서버 동시 관리
- 로그인/인증/멀티유저
- 영구 저장소(DB), 히스토리 장기 보관 — 메모리 내 최근 N분만 유지
- Docker/K8s 배포 자동화

## 4. 기술 스택

- **백엔드**: Python 3.11+, FastAPI, `prometheus_client` (텍스트 파싱용), `httpx`(폴링용), WebSocket
- **프론트**: 단일 HTML + Vanilla JS + Chart.js (React 없이 최대한 가볍게, 빌드 스텝 없는 구조 우선)
- **패키지 관리**: `uv` 또는 `venv` + `pip`
- **저장소**: 없음. 프로세스 메모리 내 `deque`로 최근 N분 데이터만 유지 (재시작하면 초기화되는 걸 의도된 동작으로 간주)

## 5. Mock 메트릭 서버 (개발용, 반드시 먼저 만들 것)

실제 vLLM 없이 개발하기 위해, `/metrics`를 Prometheus 포맷으로 반환하는 가짜 FastAPI 서버를 별도로 만든다.
- KV 캐시 사용률(`vllm:kv_cache_usage_perc`)은 사인파 + 랜덤노이즈로 0~1 사이를 오가게 시뮬레이션
- 가끔(예: 5% 확률) 90% 이상으로 스파이크를 내서 경고 UI를 테스트할 수 있게 함
- 실행중/대기중 요청 수(`vllm:num_requests_running`, `vllm:num_requests_waiting`), 토큰 카운터(`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`)도 그럴듯하게 증가시킴
- 목적: 실제 vLLM 출력 형식(라벨 포함, 위 섹션 3에서 검증된 지표 이름)만 그대로 흉내내면 됨 (값의 정확도는 중요하지 않음)
- `vllm_metrics_sample.txt`에 있는 실제 `# HELP`/`# TYPE` 주석 줄까지 그대로 복사해서 쓰면 진짜 파서 호환성 테스트가 더 정확해짐

## 6. 프로젝트 구조 (제안)

```
vllm-ops-dashboard/
├── CLAUDE.md                 # 이 문서
├── backend/
│   ├── main.py                # FastAPI 앱, WebSocket 엔드포인트
│   ├── poller.py               # /metrics 폴링 + 파싱 로직
│   ├── store.py                 # 메모리 내 최근 N분 시계열 저장
│   └── mock_vllm_server.py       # 개발용 가짜 /metrics 서버
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── pyproject.toml (or requirements.txt)
└── README.md
```

## 7. 개발 순서 (Claude Code에게 요청할 작업 단위)

1. mock_vllm_server.py 작성 — `/metrics`에서 위 6개 지표를 Prometheus 텍스트 포맷으로 반환
2. poller.py — mock 서버를 대상으로 폴링 + 파싱이 정확히 되는지 유닛테스트 수준 확인
3. store.py — 최근 N분 인메모리 저장 구조
4. main.py — FastAPI + WebSocket으로 프론트에 실시간 전달
5. frontend — 카드 4개 + 차트 붙이기
6. (선택) 맥에 실제 vLLM 설치해서 mock 대신 실제 서버로 1회 검증

## 8. 참고: vLLM 공식 메트릭 문서
- https://docs.vllm.ai/en/stable/design/metrics/
- 메트릭은 vLLM 버전마다 이름/구성이 바뀔 수 있으므로, 실제 붙일 때는 `curl <서버>/metrics`로 원본을 직접 확인해서 6번 지표 이름이 맞는지 재확인할 것.
