# bybit_ws_latency_profiler.py
"""Measure end‑to‑end latency (exchange timestamp → local receipt) for Bybit public trades via WebSocket.

The script launches several parallel WebSocket clients that all subscribe to the same `publicTrade.{SYMBOL}`
stream.  For each trade it records:
    • the exchange‑side matching timestamp embedded in the message (T)
    • the local receipt timestamp when the message hits the client

The exchange and local clocks are synchronised first via the `/v5/market/time` REST endpoint so that the
measured delay ≈ network + Bybit’s internal dissemination delay + client processing.

The results are summarised per client and a Plotly box‑plot is generated.

Requirements:
    pip install websocket-client requests plotly pandas numpy
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from multiprocessing import Event, Process, Queue
from queue import Empty
from typing import Deque, Tuple

import numpy
import pandas as pd
import plotly.express as px
import requests
import websocket

# ------------------- Configuration ------------------- #
SYMBOL = "BTCUSDT"               # Trading pair
CATEGORY = "spot"                # public WS category: spot | linear | inverse | option | spread
NUM_CLIENTS = 10                 # parallel WS clients
TRADES_SAMPLE_SIZE = 2_000       # stop after this many trades
CLOCK_SYNC_SAMPLES = 15          # how many samples to median‑filter clock offset
WS_ENDPOINT = f"wss://stream.bybit.com/v5/public/{CATEGORY}"

# ----------------------------------------------------- #


def _make_sub_msg(symbol: str) -> str:
    """Bybit WS subscribe message."""
    return json.dumps({
        "op": "subscribe",
        "args": [f"publicTrade.{symbol.upper()}"]
    })


class TimeSynchronizer:
    def __init__(self):
        self._time_offset_ms_queue: Deque[float] = deque(maxlen=CLOCK_SYNC_SAMPLES)
        self._time_offset_ms = -1
        self._lock = asyncio.Lock()

    @property
    def time(self) -> float:
        return self._current_seconds_counter() + self.time_offset_ms * 1e-3

    @property
    def time_offset_ms(self) -> float:
        if len(self._time_offset_ms_queue) < self._time_offset_ms_queue.maxlen:
            raise RuntimeError("Not initialized yet")
        elif self._time_offset_ms == -1:
            median = numpy.median(self._time_offset_ms_queue)
            weighted_average = numpy.average(
                self._time_offset_ms_queue, weights=range(1, len(self._time_offset_ms_queue) * 2 + 1, 2)
            )
            self._time_offset_ms = numpy.mean([median, weighted_average])

        return self._time_offset_ms

    def initialize(self):
        print("Synchronizing time with Bybit")
        for _ in range(self._time_offset_ms_queue.maxlen):
            self._update_server_time_offset_with_time_provider()
            time.sleep(2)
        print(f"local time = {time.time()}:: server time = {self.time}")  # to compute the offset

    def _update_server_time_offset_with_time_provider(self):
        local_before_ms: float = self._current_seconds_counter() * 1e3
        server_time_ms: float = self._fetch_bybit_time_ms()
        local_after_ms: float = self._current_seconds_counter() * 1e3
        local_server_time_pre_image_ms: float = (local_before_ms + local_after_ms) / 2.0
        time_offset_ms: float = server_time_ms - local_server_time_pre_image_ms
        self._add_time_offset_ms_sample(time_offset_ms)

    @staticmethod
    def _fetch_bybit_time_ms() -> float:
        url = "https://api.bybit.com/v5/market/time"
        data = requests.get(url, timeout=3).json()
        # Top‑level field `time` is server ms according to doc example
        return float(data.get("time"))

    def _add_time_offset_ms_sample(self, offset: float):
        self._time_offset_ms_queue.append(offset)

    @staticmethod
    def _current_seconds_counter():
        return time.time()


class WSClient(Process):
    """Single WebSocket client running in its own thread."""
    client_type = "ws"

    def __init__(
        self,
        client_id: int,
        symbol: str,
        out_queue: Queue,
        stop_event: Event,
        time_offset: float,
    ):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ws: websocket.WebSocketApp | None = None
        self.time_offset = time_offset

    def run(self):
        def on_open(ws):
            ws.send(_make_sub_msg(symbol=SYMBOL))

        def on_message(ws, message):
            recv_ts = time.time() + self.time_offset
            try:
                msg = json.loads(message)
                for t in msg.get("data", []):
                    trade_id: str = t.get("i")
                    event_ts = t.get("T", 0) / 1e3  # convert ms→s
                    self.out_queue.put_nowait((trade_id, event_ts, self.client_id, recv_ts, self.client_type))
            except (json.JSONDecodeError, KeyError, TypeError):
                return  # ignore malformed lines

        def on_error(ws, error):
            print(f"[Client {self.client_id}] error: {error}")

        self.ws = websocket.WebSocketApp(
            WS_ENDPOINT,
            on_open=on_open,
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


class TradeAggregator:
    """Collects per‑trade timestamps until all clients have reported."""

    def __init__(self, expected_clients: int):
        self.expected = expected_clients
        self._buffer: dict[str, dict[int, Tuple[float, float, str]]] = defaultdict(dict)

    def add(self, trade_id: str, event_ts: float, client_id: int, recv_ts: float, client_type: str):
        """Add a receipt; return full dict if complete, else None."""
        self._buffer[trade_id][client_id] = (event_ts, recv_ts, client_type)
        if len(self._buffer[trade_id]) == self.expected:
            return self._buffer.pop(trade_id)
        return None


# ----------------------------- main ----------------------------- #

def main():
    time_synchronizer = TimeSynchronizer()
    time_synchronizer.initialize()

    websocket.enableTrace(False)
    out_queue = Queue()
    stop_event = Event()

    # Launch clients
    ws_clients = [
        WSClient(i, SYMBOL, out_queue, stop_event, time_synchronizer.time_offset_ms * 1e-3)
        for i in range(NUM_CLIENTS)
    ]
    clients = ws_clients
    for c in clients:
        c.start()

    aggregator = TradeAggregator(expected_clients=len(clients))
    delays_per_client: dict[str, dict[int, list[float]]] = {
        WSClient.client_type: defaultdict(list),
    }
    first_counts = defaultdict(int)
    last_counts = defaultdict(int)
    ts_delay_records = []

    samples = 0
    try:
        while samples < TRADES_SAMPLE_SIZE:
            try:
                trade_id, event_ts, client_id, recv_ts, client_type = out_queue.get()
                if client_type == WSClient.client_type:
                    ts_delay_records.append({"ts": event_ts, "delay": recv_ts - event_ts})
            except Empty:
                continue
            completed = aggregator.add(trade_id, event_ts, client_id, recv_ts, client_type)
            if completed is None:
                continue  # not yet full set
            print(f"Trade {trade_id} completed.")

            # Compute per‑client delays for this trade
            # event_ts, first_recv_ts = min(completed.values(), key=lambda v: v[1])
            first_thread = min(completed, key=lambda c_: completed.get(c_)[1])
            last_thread = max(completed, key=lambda c_: completed.get(c_)[1])

            first_counts[first_thread] += 1
            last_counts[last_thread] += 1

            for cid, (event, recv, client_type) in completed.items():
                delays_per_client[client_type][cid].append((recv - event) * 1000.0)  # ms

            samples += 1
    finally:
        # Signal threads to stop and wait for them
        stop_event.set()
        for c in clients:
            c.join(timeout=2)

    # ---------------- Reporting ---------------- #
    report_ts = int(time.time())

    print("\n=== Results ===")
    print(f"\nsample size = {TRADES_SAMPLE_SIZE} trades")
    for client_type in (WSClient.client_type,):
        print(f"\n--- Client type {client_type} ---")
        sum_of_avg_delays = 0
        min_delay_per_trade = [float("inf")] * TRADES_SAMPLE_SIZE
        cids = delays_per_client[client_type].keys()
        cids = sorted(cids)
        for cid in cids:
            delays = delays_per_client[client_type][cid]
            for i, d in enumerate(delays):
                min_delay_per_trade[i] = min(min_delay_per_trade[i], d)
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            sum_of_avg_delays += avg_delay
            print(
                f"Client {cid}: samples={len(delays):5d}  avg_delay={avg_delay:7.3f} ms  "
                f"first={first_counts.get(cid, 0):5d}  last={last_counts.get(cid, 0):5d}"
            )

        ave_delay = sum_of_avg_delays / len(delays_per_client[client_type])
        print(f"\nAverage delay: {ave_delay:.3f} ms")
        print(f"Average min delay per trade: {sum(min_delay_per_trade) / TRADES_SAMPLE_SIZE:.3f} ms")

        # --------- Plotly visualisations ----------- #
        # 1. Delay distribution per client (box plot)
        rows = [
            {"client": cid, "delay_ms": d}
            for cid, delays in delays_per_client[client_type].items()
            for d in delays
        ]
        if rows:
            df = pd.DataFrame(rows)
            fig_delay = px.box(
                df,
                x="client",
                y="delay_ms",
                points="all",
                title=f"Bybit per‑trade latency for {client_type} EC2 SG clients (ave delay = {ave_delay:.3f} ms)",
                color_discrete_sequence=["red"],
            )
            fig_delay.update_layout(xaxis_title="Client ID", yaxis_title="Delay (ms)", xaxis=dict(tickmode="linear"))
            fig_delay.show()
            fig_delay.write_html(f"{report_ts}_{client_type}_delay_plot.html")


if __name__ == "__main__":
    main()
