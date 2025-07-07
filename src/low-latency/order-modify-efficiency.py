"""
latency_experiments.py  –  IBKR order-amend vs cancel/replace comparison
-----------------------------------------------------------------------

Prerequisites
-------------
pip install ib_insync pandas

TWS / IB Gateway must be running and in paper-trading mode,
with API enabled on port 7497 (the default paper port).

What the script does
--------------------
1. Subscribes to order / execution streams so that we receive
   server acks and exchange timestamps.
2. Runs three experiments (aggressive-amend, defensive-amend,
   aggressive-manual-amend) on the given stock symbol.
3. Measures for each experiment:

   - `t_local_cmd`  : local monotonic timestamp (ns) when the command is sent
   - `t_exch_ack_ns`   : exchange-side submit/replace timestamp as reported by IB
   - `latency_ms_ns`   : (t_exch_ack_ns − t_local_cmd) converted to ms

Because the absolute machine-vs-exchange clock offset is constant,
comparing these latencies lets us see which method is faster.
"""
import asyncio
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Order, util, OrderStatus

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 7497  # 7496 for live, 7497 for paper
CLIENT_ID = 1
SYMBOL = "TSLA"
INIT_PCT_A = 0.09  # 9 % below last close
AMEND_PCT_A = 0.05  # 5 % below last close
INIT_PCT_B = 0.05  # 5 % below last close
AMEND_PCT_B = 0.09  # 9 % below last close
INIT_PCT_C = 0.09  # 9 % below last close
NEW_PCT_C = 0.05  # 5 % below last close
SAMPLES_PER_EXPERIMENT = 10

# ---------------------------------------------------------------------
# CONNECT & GLOBALS
# ---------------------------------------------------------------------
ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID)

contract = Stock(SYMBOL, "SMART", "USD")
results: List[dict] = []  # rows for the final DataFrame


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
async def last_1m_close() -> float:
    # Request the most recent bar
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )

    # Get the last bar
    last_bar = bars[-1]
    return last_bar.close


def now_ns() -> int:
    return int(time.time() * 1e9)


async def wait_open_order(order_id: int) -> datetime:
    submit_dt = None

    async for event in ib.openOrderEvent.aiter():
        if event.order.orderId == order_id:
            for entry in event.log:
                if entry.status in OrderStatus.ActiveStates:
                    submit_dt = entry.time
                    break

        if submit_dt is not None:
            break

    return submit_dt


async def wait_amend_order(order_id: int) -> datetime:
    submit_dt = None
    amend_dt = None

    async for event in ib.orderStatusEvent.aiter():
        if event.order.orderId == order_id:
            submit_dt = None
            for entry in event.log:
                if entry.status in OrderStatus.ActiveStates:
                    if submit_dt is None:
                        submit_dt = entry.time
                    else:
                        amend_dt = entry.time

        if amend_dt is not None:
            break

    return amend_dt


async def wait_cancel_order(order_id: int) -> datetime:
    cancel_dt = None

    async for event in ib.openOrderEvent.aiter():
        if event.order.orderId == order_id:
            for entry in event.log:
                if entry.status in OrderStatus.DoneStates:
                    cancel_dt = entry.time
                    break

        if cancel_dt is not None:
            break

    return cancel_dt


# ---------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------
async def aggressive_order_amend():
    """Place 15 % below last close, then amend to 10 % below."""
    latencies_ms = []

    for _ in range(SAMPLES_PER_EXPERIMENT):
        last_close = await last_1m_close()
        initial_price = last_close * (1 - INIT_PCT_A)
        amended_price = last_close * (1 - AMEND_PCT_A)
        order_id = ib.client.getReqId()
        order = Order(orderId=order_id, action="BUY", totalQuantity=1,
                      orderType="LMT", lmtPrice=round(initial_price, 2))

        # 1️⃣ Submit
        ib.placeOrder(contract, order)
        await wait_open_order(order_id)

        # 2️⃣ Amend
        order.lmtPrice = round(amended_price, 2)
        t_cmd_amend = now_ns()
        ib.placeOrder(contract, order)  # same orderId → modify
        amend_dt = await wait_amend_order(order_id)

        exch_ts_ns = int(amend_dt.timestamp() * 1e9)
        latency_ms = (exch_ts_ns - t_cmd_amend) / 1e6
        latencies_ms.append(latency_ms)
        ib.cancelOrder(order)

        await asyncio.sleep(0.5)

    results.append({
        "experiment": "aggressive_order_amend",
        "latencies_ms": np.average(latencies_ms),
    })


async def defensive_order_amend():
    """Place 10 % below last close, then amend to 15 % below."""
    latencies_ms = []

    for _ in range(SAMPLES_PER_EXPERIMENT):
        last_close = await last_1m_close()
        initial_price = last_close * (1 - INIT_PCT_B)
        amended_price = last_close * (1 - AMEND_PCT_B)

        order_id = ib.client.getReqId()
        order = Order(orderId=order_id, action="BUY", totalQuantity=1,
                      orderType="LMT", lmtPrice=round(initial_price, 2))

        # 1️⃣ Submit
        ib.placeOrder(contract, order)
        await wait_open_order(order_id)

        # 2️⃣ Amend
        order.lmtPrice = round(amended_price, 2)
        t_cmd_amend = now_ns()
        ib.placeOrder(contract, order)
        amend_dt = await wait_amend_order(order_id)

        exch_ts_ns = amend_dt.timestamp() * 1e9
        latency_ms = (exch_ts_ns - t_cmd_amend) / 1e6
        latencies_ms.append(latency_ms)
        ib.cancelOrder(order)

        await asyncio.sleep(0.5)

    results.append({
        "experiment": "defensive_order_amend",
        "latencies_ms": np.average(latencies_ms),
    })


async def aggressive_manual_order_amend():
    """
    Place 15 % below last close, then *simultaneously* cancel & submit
    a new order 10 % below last close.
    """
    latencies_ms = []

    for _ in range(SAMPLES_PER_EXPERIMENT):
        last_close = await last_1m_close()
        initial_price = last_close * (1 - INIT_PCT_C)
        new_price = last_close * (1 - NEW_PCT_C)

        # Original order
        orig_id = ib.client.getReqId()
        orig = Order(orderId=orig_id, action="BUY", totalQuantity=1,
                     orderType="LMT", lmtPrice=round(initial_price, 2))

        ib.placeOrder(contract, orig)
        await wait_open_order(orig_id)

        # New order
        new_id = ib.client.getReqId()
        new_order = Order(orderId=new_id, action="BUY", totalQuantity=1,
                          orderType="LMT", lmtPrice=round(new_price, 2))

        # 1️⃣ Cancel-Replace
        t_cmd = now_ns()
        ib.cancelOrder(orig)
        ib.placeOrder(contract, new_order)

        # Wait for both cancel acknowledgement *and* new order ack
        res = await asyncio.gather(wait_open_order(orig_id), wait_open_order(new_id), return_exceptions=True)
        cancel_dt = res[0]
        open_dt = res[1]

        exch_ts_ns = max(cancel_dt.timestamp() * 1e9, open_dt.timestamp() * 1e9)
        latency_ms_ns = (exch_ts_ns - t_cmd) / 1e6
        latencies_ms.append(latency_ms_ns)

        ib.cancelOrder(new_order)

        await asyncio.sleep(0.5)

    results.append({
        "experiment": "aggressive_manual_order_amend",
        "latencies_ms": np.average(latencies_ms),
    })


# ---------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------
async def main():
    print("Subscribing to user stream…")
    ib.reqExecutions()  # any request brings us into the execution stream

    print("Running experiments…")
    await aggressive_order_amend()
    await defensive_order_amend()
    await aggressive_manual_order_amend()

    df = pd.DataFrame(results)
    df.columns = [
        "experiment", f"latencies_ms (ave. over {SAMPLES_PER_EXPERIMENT})"
    ]
    pd.options.display.float_format = "{:.3f}".format
    print("\n=== Latency results (ms) ===")
    print(df)

    print("\nFull details:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    util.startLoop()  # needed when run outside Jupyter
    ib.run(main())
