import os
import threading
import time
from collections import deque
from queue import Queue

from binance import ThreadedWebsocketManager
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

BASE = "BTC"
QUOTE = "USDC"
SYMBOL = f"{BASE}{QUOTE}"


def main():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)

    print(
        "Balance: {}={} {}={}".format(
            BASE,
            client.get_asset_balance(asset=BASE)["free"],
            QUOTE,
            client.get_asset_balance(asset=QUOTE)["free"],
        )
    )

    stop_event = threading.Event()
    public_trades_queue = Queue()
    private_trades_queue = Queue()

    public_trades_thread = threading.Thread(
        target=listen_to_public_trades, args=(stop_event, public_trades_queue)
    )
    private_trades_thread = threading.Thread(
        target=listen_to_private_trades, args=(stop_event, api_key, api_secret, private_trades_queue)
    )

    public_trades_thread.start()
    private_trades_thread.start()

    public_trades = deque(maxlen=200)
    private_trades = deque(maxlen=10)
    private_to_public_ts_deltas = list()

    try:
        while len(private_to_public_ts_deltas) != 1:
            public_trade = public_trades_queue.get()
            print(public_trade)
            public_trades.append(public_trade)
            if not private_trades_queue.empty():
                private_trade = private_trades_queue.get_nowait()
                print(private_trade)
                private_trades.append(private_trade)
            while len(private_trades) != 0 and len(public_trades) != 0:
                next_private, private_ts = private_trades.popleft()
                next_public, public_ts = public_trades.popleft()
                if next_private["t"] == next_public["t"]:
                    private_to_public_ts_deltas.append(private_ts - public_ts)
                    print(private_ts - public_ts)
                else:
                    while next_private["t"] < next_public["t"] and len(private_trades) != 0:
                        next_private, private_ts = private_trades.popleft()
                    while next_private["t"] > next_public["t"] and len(public_trades) != 0:
                        next_public, public_ts = public_trades.popleft()
                    private_trades.appendleft((next_private, private_ts))
                    public_trades.appendleft((next_public, public_ts))
    except KeyboardInterrupt:
        stop_event.set()
        public_trades_thread.join()

    for delta in private_to_public_ts_deltas:
        print(f"{delta} ns")

    print("Done")


def listen_to_public_trades(stop_event: threading.Event, public_trades_queue: Queue):
    def handle_msg(msg):
        public_trades_queue.put_nowait((msg, int(time.time() * 1e9)))

    twm = ThreadedWebsocketManager()
    twm.start()
    twm.start_trade_socket(callback=handle_msg, symbol=SYMBOL.lower())
    stop_event.wait()
    twm.stop()


def listen_to_private_trades(
    stop_event: threading.Event, api_key: str, api_secret: str, private_trades_queue: Queue
):
    def handle_msg(msg):
        ts = int(time.time() * 1e9)
        evt = msg.get("e")
        if evt == "executionReport" and msg["x"] == "TRADE":
            private_trades_queue.put_nowait((msg, ts))

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()
    twm.start_user_socket(callback=handle_msg)
    stop_event.wait()
    twm.stop()


if __name__ == "__main__":
    main()
