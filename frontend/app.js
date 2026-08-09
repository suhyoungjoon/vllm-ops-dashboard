const WINDOW_SECONDS = 600; // 최근 10분
const KV_WARNING_THRESHOLD = 0.9;

const state = {
  points: [], // { timestamp, kv, running, waiting }
  lastGenerationTokens: null,
  lastTimestamp: null,
};

const el = {
  status: document.getElementById("connection-status"),
  kvCard: document.getElementById("card-kv"),
  kvValue: document.getElementById("value-kv"),
  runningValue: document.getElementById("value-running"),
  waitingValue: document.getElementById("value-waiting"),
  tpsValue: document.getElementById("value-tps"),
};

function formatTime(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString("ko-KR", { hour12: false });
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

  state.points.push({
    timestamp: message.timestamp,
    kv: message.kv_cache_usage_perc,
    running: message.num_requests_running,
    waiting: message.num_requests_waiting,
  });
  pruneOldPoints(message.timestamp);

  updateCards(message, tps);
  updateCharts();
}

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
