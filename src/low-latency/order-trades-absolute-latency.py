import asyncio
import base64
import json
import os
import socket
import ssl
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from multiprocessing import Process, Queue, Event
from queue import Empty
from typing import Tuple, Deque, List

import numpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import simplefix
import websocket
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from dotenv import load_dotenv
from simplefix import FixParser

load_dotenv()

# ---------------- Configuration ---------------- #
SYMBOL = "BTCUSDT"  # Trading pair to subscribe to (case‑insensitive)
NUM_CLIENTS = 10  # Number of parallel clients per client type (WebSocket | FIX)
TRADES_SAMPLE_SIZE = 2000
WEBSOCKET_ENDPOINT = (
    "wss://stream.binance.com:9443/ws/{symbol}@trade"
)  # Binance public trade stream
CLOCK_SYNC_SAMPLES = 15
FIX_HOSTNAME = "fix-md.binance.com"
FIX_PORT = 9000
FIX_API_KEY = os.environ.get("BINANCE_FIX_API_KEY")
PRIVATE_KEY_PATH = os.environ.get("BINANCE_FIX_PRIVATE_KEY_PATH")
CA_CERT_PATH = os.environ.get("CA_CERT_PATH")
BASE_SENDER_COMP_ID_NUMERIC_PORTION = int(time.time()) % 1_000


# ------------------------------------------------- #


def make_ws_url(symbol: str) -> str:
    """Return the full websocket URL for the given symbol."""
    return WEBSOCKET_ENDPOINT.format(symbol=symbol.lower())


class TradeAggregator:
    """Collects per‑trade timestamps until all clients have reported."""

    def __init__(self, expected_clients: int):
        self.expected = expected_clients
        self._buffer: dict[int, dict[int, Tuple[float, float, str]]] = defaultdict(dict)

    def add(self, trade_id: int, event_ts: float, client_id: int, recv_ts: float, client_type: str):
        """Add a receipt; return full dict if complete, else None."""
        self._buffer[trade_id][client_id] = (event_ts, recv_ts, client_type)
        if len(self._buffer[trade_id]) == self.expected:
            return self._buffer.pop(trade_id)
        return None


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
        self.url = make_ws_url(symbol)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ws: websocket.WebSocketApp | None = None
        self.time_offset = time_offset

    def run(self):
        def on_message(ws, message):
            recv_ts = time.time() + self.time_offset
            try:
                data = json.loads(message)
                trade_id = data["t"]
                event_ts = data["E"] * 1e-3
            except (KeyError, json.JSONDecodeError):
                return  # skip malformed
            self.out_queue.put_nowait((trade_id, event_ts, self.client_id, recv_ts, self.client_type))

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


class FIXClient(Process):
    client_type = "fix"

    def __init__(
        self,
        client_id: int,
        symbol: str,
        out_queue: Queue,
        stop_event: Event,
        time_offset: float,
    ):
        super().__init__(daemon=False)
        self.client_id = client_id
        self.symbol = symbol
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.time_offset = time_offset
        self.sender_comp_id = f"TR{BASE_SENDER_COMP_ID_NUMERIC_PORTION + client_id}"
        self.seq = 1

    def run(self):
        with open(PRIVATE_KEY_PATH, "rb") as f:
            private_key = load_pem_private_key(f.read(), password=None)

        sending_time = (
            datetime.fromtimestamp(
                time.time() + self.time_offset, tz=timezone.utc
            ).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        )
        raw_data = self.sign_logon(private_key=private_key, target_comp_id="SPOT", time_=sending_time)

        context = ssl.create_default_context()
        context.load_verify_locations(cafile=CA_CERT_PATH)
        parser = FixParser()

        with socket.create_connection((FIX_HOSTNAME, FIX_PORT)) as sock:
            with context.wrap_socket(sock, server_hostname=FIX_HOSTNAME) as ssock:
                logon_msg = self.build_logon(sending_time, raw_data)
                ssock.sendall(logon_msg.encode())

                time.sleep(1)
                self.seq += 1
                sub_msg = self.build_trade_subscription()
                ssock.sendall(sub_msg.encode())

                buffer = b""
                while not self.stop_event.is_set():
                    data = ssock.recv(4096)
                    recv_ts = time.time() + self.time_offset
                    buffer += data
                    parser.append_buffer(data)
                    while True:
                        msg = parser.get_message()
                        if msg is None:
                            break
                        self.process_msg(msg=msg, recv_ts=recv_ts, ssock=ssock)

    def sign_logon(self, private_key: Ed25519PrivateKey, target_comp_id: str, time_: str):
        payload = chr(1).join(["A", self.sender_comp_id, target_comp_id, str(self.seq), time_])
        sig = private_key.sign(payload.encode("ASCII"))
        return base64.b64encode(sig).decode("ASCII")

    def build_logon(self, sending_time, raw_data):
        msg = simplefix.FixMessage()
        msg.append_string("8=FIX.4.4", header=True)
        msg.append_pair(35, "A", header=True)
        msg.append_pair(34, self.seq, header=True)
        msg.append_pair(49, self.sender_comp_id, header=True)
        msg.append_pair(52, sending_time, header=True)
        msg.append_pair(56, "SPOT", header=True)
        msg.append_pair(95, len(raw_data))
        msg.append_pair(96, raw_data)
        msg.append_pair(98, 0)
        msg.append_pair(108, 30)
        msg.append_pair(141, "Y")
        msg.append_pair(553, FIX_API_KEY)
        msg.append_pair(25035, 1)  # UNORDERED
        return msg

    def build_trade_subscription(self):
        msg = simplefix.FixMessage()
        msg.append_string("8=FIX.4.4", header=True)
        msg.append_pair(35, "V", header=True)
        msg.append_pair(34, self.seq, header=True)
        msg.append_pair(49, self.sender_comp_id, header=True)
        msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3], header=True)
        msg.append_pair(56, "SPOT", header=True)
        msg.append_pair(262, "TRADE_STREAM")
        msg.append_pair(263, 1)  # SUBSCRIBE
        msg.append_pair(264, 1)
        msg.append_pair(266, "Y")
        msg.append_pair(146, 1)
        msg.append_pair(55, self.symbol)
        msg.append_pair(267, 1)
        msg.append_pair(269, 2)  # TRADE
        return msg

    def process_msg(self, msg: simplefix.FixMessage, recv_ts: float, ssock: socket.socket):
        current_entry = {}
        in_trade_entry = False

        for tag, value in msg.pairs:
            tag = int(tag)
            val = value.decode("ascii") if isinstance(value, bytes) else value

            if tag == 35 and val == "1":
                self.reply_with_heart_beat(fields=msg.pairs, ssock=ssock)
            if tag == 269 and val == "2":  # TRADE
                in_trade_entry = True
            elif tag == 1003:  # Trade ID
                current_entry["trade_id"] = int(val)
            elif tag == 60:  # TransactTime
                current_entry["event_ts"] = (
                    datetime.strptime(val + " +0000", "%Y%m%d-%H:%M:%S.%f %z").timestamp()
                )
            elif tag == 279:  # Start of a new MDEntry
                if in_trade_entry and "trade_id" in current_entry:
                    self.out_queue.put_nowait(
                        (current_entry["trade_id"], current_entry["event_ts"], self.client_id, recv_ts,
                         self.client_type)
                    )
                current_entry = {}
                in_trade_entry = False

        # Capture last entry if needed
        if in_trade_entry and "trade_id" in current_entry:
            self.out_queue.put_nowait(
                (current_entry["trade_id"], current_entry["event_ts"], self.client_id, recv_ts, self.client_type)
            )

    def reply_with_heart_beat(self, fields: List[Tuple[int, bytes]], ssock: socket.socket):
        try:
            test_req_id = next(f[1] for f in fields if int(f[0]) == 112)
        except StopIteration:
            test_req_id = b''
        test_req_id = test_req_id.decode()
        heartbeat = simplefix.FixMessage()
        heartbeat.append_string("8=FIX.4.4", header=True)
        self.seq += 1
        heartbeat.append_pair(35, "0", header=True)
        heartbeat.append_pair(34, self.seq, header=True)
        heartbeat.append_pair(49, self.sender_comp_id, header=True)
        heartbeat.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3], header=True)
        heartbeat.append_pair(56, "SPOT", header=True)
        heartbeat.append_pair(112, test_req_id)
        ssock.sendall(heartbeat.encode())
        print(f"✅ Sent Heartbeat in response to TestRequest: {test_req_id}")


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
        print("Synchronizing time with Binance")
        for _ in range(self._time_offset_ms_queue.maxlen):
            self._update_server_time_offset_with_time_provider()
            time.sleep(2)
        print(f"local time = {time.time()}:: server time = {self.time}")  # to compute the offset

    def _update_server_time_offset_with_time_provider(self):
        local_before_ms: float = self._current_seconds_counter() * 1e3
        server_time_ms: float = self._get_binance_time_ms()
        local_after_ms: float = self._current_seconds_counter() * 1e3
        local_server_time_pre_image_ms: float = (local_before_ms + local_after_ms) / 2.0
        time_offset_ms: float = server_time_ms - local_server_time_pre_image_ms
        self._add_time_offset_ms_sample(time_offset_ms)

    @staticmethod
    def _get_binance_time_ms():
        url = "https://api.binance.com/api/v3/time"
        response = requests.get(url)
        data = response.json()
        server_time_ms = data["serverTime"]
        return server_time_ms

    def _add_time_offset_ms_sample(self, offset: float):
        self._time_offset_ms_queue.append(offset)

    @staticmethod
    def _current_seconds_counter():
        return time.time()


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
    fix_clients = [
        FIXClient(i, SYMBOL, out_queue, stop_event, time_synchronizer.time_offset_ms * 1e-3)
        for i in range(NUM_CLIENTS, NUM_CLIENTS * 2)
    ]
    clients = ws_clients + fix_clients
    for c in clients:
        c.start()

    aggregator = TradeAggregator(expected_clients=len(ws_clients) + len(fix_clients))
    delays_per_client: dict[str, dict[int, list[float]]] = {
        WSClient.client_type: defaultdict(list),
        FIXClient.client_type: defaultdict(list),
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
    for client_type in (WSClient.client_type, FIXClient.client_type):
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
            fig_delay = px.box(df, x="client", y="delay_ms", points="all",
                               title=f"Per‑trade latency for {client_type} local machine clients (ave delay = {ave_delay:.3f} ms)")
            fig_delay.update_layout(xaxis_title="Client ID", yaxis_title="Delay (ms)", xaxis=dict(tickmode="linear"))
            fig_delay.show()
            fig_delay.write_html(f"{report_ts}_{client_type}_delay_plot.html")

    # ts_delay_df = pd.DataFrame(ts_delay_records)
    # bin_width_ms = 100
    # if not ts_delay_df.empty:
    #     # Convert timestamps to bin indices
    #     t0 = ts_delay_df["ts"].min()
    #     ts_delay_df["bin"] = ((ts_delay_df["ts"] - t0) * 1000 // bin_width_ms).astype(int)
    #
    #     # Group by time bin
    #     grouped = ts_delay_df.groupby("bin").agg(
    #         order_count=("ts", "count"),
    #         ave_delay_ms=("delay", "mean"),
    #         start_time=("ts", "min"),
    #     ).reset_index()
    #
    #     # Plot: order count vs average latency
    #     fig = px.scatter(
    #         grouped,
    #         x="order_count",
    #         y="ave_delay_ms",
    #         title="Order Volume vs Average Latency (Binned 200ms)",
    #         labels={
    #             "order_count": "Orders in 200ms Window",
    #             "ave_delay_ms": "Average Delay (ms)"
    #         },
    #     )
    #     fig.show()
    #
    #     ts_delay_df["bin"] = ((ts_delay_df["ts"] - t0) * 1000 // bin_width_ms).astype(int)
    #     delay_over_time = ts_delay_df.groupby("bin").agg(
    #         ave_delay_ms=("delay", "mean"),
    #         start_time=("ts", "min")
    #     ).reset_index()
    #
    #     fig_time = px.line(
    #         delay_over_time,
    #         x="start_time",
    #         y="ave_delay_ms",
    #         title="Average Latency Over Time (100ms bins)",
    #         labels={"start_time": "Time", "ave_delay_ms": "Average Delay (ms)"}
    #     )
    #     fig_time.update_traces(mode="lines+markers")
    #     fig_time.show()


if __name__ == "__main__":
    main()
