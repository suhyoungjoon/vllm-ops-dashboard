const WINDOW_SECONDS = 600; // 최근 10분
const KV_WARNING_THRESHOLD = 0.9;

const state = {
  points: [], // { timestamp, kv, running, waiting, tokens }
  lastGenerationTokens: null,
  lastTimestamp: null,
  latestMessage: null,
};

const el = {
  status: document.getElementById("connection-status"),
  downloadReportButton: document.getElementById("download-report"),
  kvCard: document.getElementById("card-kv"),
  kvValue: document.getElementById("value-kv"),
  runningValue: document.getElementById("value-running"),
  waitingValue: document.getElementById("value-waiting"),
  tpsValue: document.getElementById("value-tps"),
  ttftP50: document.getElementById("value-ttft-p50"),
  ttftP90: document.getElementById("value-ttft-p90"),
  ttftP99: document.getElementById("value-ttft-p99"),
  e2eP50: document.getElementById("value-e2e-p50"),
  e2eP90: document.getElementById("value-e2e-p90"),
  e2eP99: document.getElementById("value-e2e-p99"),
  queueTime: document.getElementById("value-queue-time"),
  inferenceTime: document.getElementById("value-inference-time"),
  preemptions: document.getElementById("value-preemptions"),
  prefixHitRate: document.getElementById("value-prefix-hit-rate"),
  concurrency: document.getElementById("value-concurrency"),
  saturationEta: document.getElementById("value-saturation-eta"),
  costPromptTokens: document.getElementById("value-cost-prompt-tokens"),
  costGenerationTokens: document.getElementById("value-cost-generation-tokens"),
  costEstimate: document.getElementById("value-cost-estimate"),
  configPanel: document.getElementById("panel-config"),
  configGrid: document.getElementById("config-grid"),
  diagnosisPanel: document.getElementById("diagnosis-panel"),
};

const DIAGNOSIS_ICONS = { warning: "⚠", info: "ℹ", ok: "✓" };

function renderDiagnosis(diagnosis) {
  if (!diagnosis || diagnosis.length === 0) return;
  el.diagnosisPanel.innerHTML = diagnosis
    .map((d) => {
      const recommendation = d.recommendation
        ? `<div class="diagnosis-item__recommendation">→ ${d.recommendation}</div>`
        : "";
      return `<div class="diagnosis-item diagnosis-item--${d.level}"><span class="diagnosis-item__icon">${DIAGNOSIS_ICONS[d.level] || ""}</span><div><div>${d.message}</div>${recommendation}</div></div>`;
    })
    .join("");
}

// 서빙 설정(vllm:cache_config_info) 라벨 중 사용자에게 의미 있는 항목만 골라 보여준다.
const CONFIG_FIELDS = [
  ["cache_dtype", "KV 캐시 dtype"],
  ["block_size", "블록 크기"],
  ["num_gpu_blocks", "GPU 블록 수"],
  ["gpu_memory_utilization", "GPU 메모리 사용률 설정"],
  ["kv_cache_max_concurrency", "KV 캐시 최대 동시성"],
  ["enable_prefix_caching", "Prefix 캐싱"],
  ["prefix_caching_hash_algo", "Prefix 해시 알고리즘"],
];

function formatTime(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString("ko-KR", { hour12: false });
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return "--";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(2)}s`;
}

function formatPercent(ratio) {
  return ratio === null || ratio === undefined ? "--" : `${(ratio * 100).toFixed(1)}%`;
}

function renderConfigPanel(cacheConfig) {
  if (!cacheConfig || el.configPanel.dataset.rendered) return;
  el.configGrid.innerHTML = CONFIG_FIELDS.filter(([key]) => key in cacheConfig)
    .map(
      ([key, label]) =>
        `<div class="config-item"><span class="config-item__label">${label}</span><span class="config-item__value">${cacheConfig[key]}</span></div>`
    )
    .join("");
  el.configPanel.hidden = false;
  el.configPanel.dataset.rendered = "true";
}

function formatConcurrency(capacity) {
  if (!capacity) return "--";
  if (capacity.max_concurrency === null || capacity.max_concurrency === undefined) {
    return capacity.current_concurrency.toFixed(0);
  }
  return `${capacity.current_concurrency.toFixed(0)} / ${capacity.max_concurrency.toFixed(1)}`;
}

function formatSaturationEta(capacity) {
  if (!capacity || capacity.kv_trend_per_minute === null || capacity.kv_trend_per_minute === undefined) {
    return "데이터 수집 중";
  }
  if (capacity.minutes_to_saturation === null || capacity.minutes_to_saturation === undefined) {
    return "안정적";
  }
  if (capacity.minutes_to_saturation <= 0) return "임박";
  return `약 ${capacity.minutes_to_saturation.toFixed(1)}분 후`;
}

function formatTokenCount(n) {
  return n === null || n === undefined ? "--" : `${Math.round(n).toLocaleString("ko-KR")}`;
}

function formatCostEstimate(cost) {
  if (!cost || !cost.enabled) return "단가 미설정";
  return `$${cost.estimated_cost_usd.toFixed(4)}`;
}

function updateAdvancedStats(message) {
  el.ttftP50.textContent = formatSeconds(message.ttft_p50_seconds);
  el.ttftP90.textContent = formatSeconds(message.ttft_p90_seconds);
  el.ttftP99.textContent = formatSeconds(message.ttft_p99_seconds);
  el.e2eP50.textContent = formatSeconds(message.e2e_p50_seconds);
  el.e2eP90.textContent = formatSeconds(message.e2e_p90_seconds);
  el.e2eP99.textContent = formatSeconds(message.e2e_p99_seconds);
  el.queueTime.textContent = formatSeconds(message.queue_time_avg_seconds);
  el.inferenceTime.textContent = formatSeconds(message.inference_time_avg_seconds);
  el.preemptions.textContent = message.num_preemptions_total.toFixed(0);
  el.prefixHitRate.textContent = formatPercent(message.prefix_cache_hit_rate);
  el.concurrency.textContent = formatConcurrency(message.capacity);
  el.saturationEta.textContent = formatSaturationEta(message.capacity);
  el.costPromptTokens.textContent = formatTokenCount(message.prompt_tokens_total);
  el.costGenerationTokens.textContent = formatTokenCount(message.generation_tokens_total);
  el.costEstimate.textContent = formatCostEstimate(message.cost);
  renderConfigPanel(message.cache_config);
}

function pruneOldPoints(latestTimestamp) {
  const cutoff = latestTimestamp - WINDOW_SECONDS;
  while (state.points.length && state.points[0].timestamp < cutoff) {
    state.points.shift();
  }
}

const kvChart = new Chart(document.getElementById("chart-kv"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "KV 캐시 사용률 (%)",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.15)",
        tension: 0.25,
        pointRadius: 0,
        fill: true,
      },
    ],
  },
  options: {
    animation: false,
    scales: { y: { min: 0, max: 100 } },
  },
});

const requestsChart = new Chart(document.getElementById("chart-requests"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "실행중",
        data: [],
        borderColor: "#22c55e",
        pointRadius: 0,
        tension: 0.25,
      },
      {
        label: "대기중",
        data: [],
        borderColor: "#f59e0b",
        pointRadius: 0,
        tension: 0.25,
      },
    ],
  },
  options: {
    animation: false,
    scales: { y: { min: 0 } },
  },
});

function computeTps(message) {
  if (state.lastGenerationTokens === null) return null;
  const deltaTokens = message.generation_tokens_total - state.lastGenerationTokens;
  const deltaTime = message.timestamp - state.lastTimestamp;
  if (deltaTime <= 0) return null;
  return Math.max(0, deltaTokens / deltaTime);
}

function updateCards(message, tps) {
  const kvPercent = message.kv_cache_usage_perc * 100;
  el.kvValue.textContent = `${kvPercent.toFixed(1)}%`;
  el.runningValue.textContent = message.num_requests_running.toFixed(0);
  el.waitingValue.textContent = message.num_requests_waiting.toFixed(0);
  el.tpsValue.textContent = tps === null ? "--" : tps.toFixed(1);

  el.kvCard.classList.toggle("card--warning", message.kv_cache_usage_perc > KV_WARNING_THRESHOLD);
}

function updateCharts() {
  const labels = state.points.map((p) => formatTime(p.timestamp));

  kvChart.data.labels = labels;
  kvChart.data.datasets[0].data = state.points.map((p) => p.kv * 100);
  kvChart.update("none");

  requestsChart.data.labels = labels;
  requestsChart.data.datasets[0].data = state.points.map((p) => p.running);
  requestsChart.data.datasets[1].data = state.points.map((p) => p.waiting);
  requestsChart.update("none");
}

function handleMessage(message) {
  const tps = computeTps(message);
  state.lastGenerationTokens = message.generation_tokens_total;
  state.lastTimestamp = message.timestamp;
  state.latestMessage = message;

  state.points.push({
    timestamp: message.timestamp,
    kv: message.kv_cache_usage_perc,
    running: message.num_requests_running,
    waiting: message.num_requests_waiting,
    tokens: message.generation_tokens_total,
  });
  pruneOldPoints(message.timestamp);

  updateCards(message, tps);
  updateCharts();
  updateAdvancedStats(message);
  renderDiagnosis(message.diagnosis);
}

function computeWindowStats() {
  if (state.points.length === 0) return null;
  const first = state.points[0];
  const last = state.points[state.points.length - 1];
  const kvValues = state.points.map((p) => p.kv);
  const durationSeconds = last.timestamp - first.timestamp;
  const tokenDelta = last.tokens - first.tokens;
  const avgTps = durationSeconds > 0 ? Math.max(0, tokenDelta / durationSeconds) : null;

  return {
    count: state.points.length,
    startTimestamp: first.timestamp,
    endTimestamp: last.timestamp,
    durationSeconds,
    avgKv: kvValues.reduce((a, b) => a + b, 0) / kvValues.length,
    maxKv: Math.max(...kvValues),
    minKv: Math.min(...kvValues),
    avgTps,
  };
}

function buildReportMarkdown() {
  const msg = state.latestMessage;
  const win = computeWindowStats();
  if (!msg || !win) {
    return "# vLLM Ops 요약 리포트\n\n아직 수집된 데이터가 없습니다.\n";
  }

  const lines = [
    "# vLLM Ops 요약 리포트",
    "",
    `생성 시각: ${new Date().toLocaleString("ko-KR")}`,
    `집계 구간: ${formatTime(win.startTimestamp)} ~ ${formatTime(win.endTimestamp)} (약 ${(win.durationSeconds / 60).toFixed(1)}분, 스냅샷 ${win.count}개)`,
    "",
    "## 처리량",
    `- 평균 TPS: ${win.avgTps === null ? "--" : win.avgTps.toFixed(1)} 토큰/초`,
    `- 누적 입력 토큰: ${formatTokenCount(msg.prompt_tokens_total)}`,
    `- 누적 출력 토큰: ${formatTokenCount(msg.generation_tokens_total)}`,
    "",
    "## KV 캐시 사용률 (집계 구간)",
    `- 평균: ${(win.avgKv * 100).toFixed(1)}%`,
    `- 최고: ${(win.maxKv * 100).toFixed(1)}%`,
    `- 최저: ${(win.minKv * 100).toFixed(1)}%`,
    "",
    "## 지연시간 (최근 시점 기준)",
    `- TTFT p50 / p90 / p99: ${formatSeconds(msg.ttft_p50_seconds)} / ${formatSeconds(msg.ttft_p90_seconds)} / ${formatSeconds(msg.ttft_p99_seconds)}`,
    `- 전체 요청 지연 p50 / p90 / p99: ${formatSeconds(msg.e2e_p50_seconds)} / ${formatSeconds(msg.e2e_p90_seconds)} / ${formatSeconds(msg.e2e_p99_seconds)}`,
    "",
    "## 운영 신호",
    `- 대기시간(평균): ${formatSeconds(msg.queue_time_avg_seconds)}`,
    `- 처리시간(평균): ${formatSeconds(msg.inference_time_avg_seconds)}`,
    `- 선점 누계: ${msg.num_preemptions_total.toFixed(0)}`,
    `- Prefix 캐시 적중률: ${formatPercent(msg.prefix_cache_hit_rate)}`,
    `- 동시성 여유: ${formatConcurrency(msg.capacity)}`,
    `- 포화 예상까지: ${formatSaturationEta(msg.capacity)}`,
    "",
    "## 현재 진단",
    ...(msg.diagnosis || []).map(
      (d) => `- [${d.level}] ${d.message}${d.recommendation ? ` → ${d.recommendation}` : ""}`
    ),
    "",
    "## 비용 참고",
    `- 상용 API 환산 비용: ${formatCostEstimate(msg.cost)}`,
  ];

  if (msg.cache_config) {
    lines.push("", "## 서빙 설정");
    for (const [key, label] of CONFIG_FIELDS) {
      if (key in msg.cache_config) lines.push(`- ${label}: ${msg.cache_config[key]}`);
    }
  }

  lines.push("");
  return lines.join("\n");
}

function downloadReport() {
  const markdown = buildReportMarkdown();
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `vllm-ops-report-${stamp}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

el.downloadReportButton.addEventListener("click", downloadReport);

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);

  ws.addEventListener("open", () => {
    el.status.textContent = "연결됨";
    el.status.className = "status status--connected";
    // 재연결 시 서버가 백로그를 처음부터 다시 보내므로 클라이언트 상태도 초기화한다
    state.points = [];
    state.lastGenerationTokens = null;
    state.lastTimestamp = null;
  });

  ws.addEventListener("message", (event) => {
    handleMessage(JSON.parse(event.data));
  });

  ws.addEventListener("close", () => {
    el.status.textContent = "재연결 중...";
    el.status.className = "status status--connecting";
    setTimeout(connect, 2000);
  });

  ws.addEventListener("error", () => ws.close());
}

connect();
