import json
import threading
import time
from collections import defaultdict
from queue import Queue, Empty

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import websocket

# ---------------- Configuration ---------------- #
SYMBOL = "BTCUSDT"  # Trading pair to subscribe to (case‑insensitive)
NUM_CLIENTS = 20  # Number of parallel WebSocket clients
DURATION_SECONDS = 120  # How long to capture (in seconds)
WEBSOCKET_ENDPOINT = (
    "wss://stream.binance.com:9443/ws/{symbol}@trade"
)  # Binance public trade stream


# ------------------------------------------------- #


def make_ws_url(symbol: str) -> str:
    """Return the full websocket URL for the given symbol."""
    return WEBSOCKET_ENDPOINT.format(symbol=symbol.lower())


class TradeAggregator:
    """Collects per‑trade timestamps until all clients have reported."""

    def __init__(self, expected_clients: int):
        self.expected = expected_clients
        self._buffer: dict[int, dict[int, float]] = defaultdict(dict)

    def add(self, trade_id: int, client_id: int, ts: float):
        """Add a receipt; return full dict if complete, else None."""
        self._buffer[trade_id][client_id] = ts
        if len(self._buffer[trade_id]) == self.expected:
            return self._buffer.pop(trade_id)
        return None


class WSClient(threading.Thread):
    """Single WebSocket client running in its own thread."""

    def __init__(self, client_id: int, symbol: str, out_queue: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.url = make_ws_url(symbol)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ws: websocket.WebSocketApp | None = None

    def run(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                trade_id = data["t"]
            except (KeyError, json.JSONDecodeError):
                return  # skip malformed
            recv_ts = time.time()
            self.out_queue.put_nowait((trade_id, self.client_id, recv_ts))

        def on_error(ws, error):
            print(f"[Client {self.client_id}] error: {error}")

        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_error=on_error,
        )

        # Run until stop_event is set; ping_interval keeps the connection alive.
        while not self.stop_event.is_set():
            try:
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[Client {self.client_id}] run_forever exception: {e}")
                time.sleep(1)  # brief back‑off before reconnect

        # Graceful shutdown
        try:
            self.ws.close()
        except Exception:
            pass


def main():
    websocket.enableTrace(False)
    out_queue: Queue = Queue()
    stop_event = threading.Event()

    # Launch clients
    clients = [WSClient(i, SYMBOL, out_queue, stop_event) for i in range(NUM_CLIENTS)]
    for c in clients:
        c.start()

    aggregator = TradeAggregator(expected_clients=NUM_CLIENTS)
    delays_per_client: dict[int, list[float]] = defaultdict(list)
    first_counts = defaultdict(int)
    last_counts = defaultdict(int)

    start_time = time.time()
    try:
        while time.time() - start_time < DURATION_SECONDS:
            try:
                trade_id, client_id, ts = out_queue.get()
            except Empty:
                continue
            completed = aggregator.add(trade_id, client_id, ts)
            if completed is None:
                continue  # not yet full set

            # Compute per‑client delays for this trade
            first_ts = min(completed.values())
            first_thread = min(completed, key=completed.get)
            last_thread = max(completed, key=completed.get)

            first_counts[first_thread] += 1
            last_counts[last_thread] += 1

            for cid, recv in completed.items():
                delays_per_client[cid].append((recv - first_ts) * 1000.0)  # ms
    finally:
        # Signal threads to stop and wait for them
        stop_event.set()
        for c in clients:
            c.join(timeout=2)

    # ---------------- Reporting ---------------- #
    print("\n=== Results ===")
    sum_of_avg_delays = 0
    for cid in range(NUM_CLIENTS):
        delays = delays_per_client.get(cid, [])
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        sum_of_avg_delays += avg_delay
        print(
            f"Client {cid}: samples={len(delays):5d}  avg_delay={avg_delay:7.3f} ms  "
            f"first={first_counts.get(cid, 0):5d}  last={last_counts.get(cid, 0):5d}"
        )

    print(f"\nAverage delay: {sum_of_avg_delays / NUM_CLIENTS:.3f} ms")

    report_ts = int(time.time())

    # --------- Plotly visualisations ----------- #
    # 1. Delay distribution per client (box plot)
    rows = [
        {"client": cid, "delay_ms": d}
        for cid, delays in delays_per_client.items()
        for d in delays
    ]
    if rows:
        df = pd.DataFrame(rows)
        fig_delay = px.box(df, x="client", y="delay_ms", points="all",
                           title="Per‑trade latency relative to first receipt")
        fig_delay.update_layout(xaxis_title="Client ID", yaxis_title="Delay (ms)", xaxis=dict(tickmode='linear'))
        fig_delay.show()
        fig_delay.write_html(f"{report_ts}_delay_plot.html")

    # 2. First / last counts per client (stacked bar)
    bar_df = pd.DataFrame({
        "client": list(range(NUM_CLIENTS)),
        "first": [first_counts.get(cid, 0) for cid in range(NUM_CLIENTS)],
        "last": [last_counts.get(cid, 0) for cid in range(NUM_CLIENTS)],
    })
    if not bar_df.empty:
        fig_rank = go.Figure()
        fig_rank.add_bar(x=bar_df["client"], y=bar_df["first"], name="First")
        fig_rank.add_bar(x=bar_df["client"], y=bar_df["last"], name="Last")
        fig_rank.update_layout(
            barmode="group",
            title="How often each client was first vs. last per trade",
            xaxis_title="Client ID",
            yaxis_title="Count",
            xaxis=dict(tickmode='linear'),
        )
        fig_rank.show()
        fig_rank.write_html(f"{report_ts}_rank_plot.html")


if __name__ == "__main__":
    main()
