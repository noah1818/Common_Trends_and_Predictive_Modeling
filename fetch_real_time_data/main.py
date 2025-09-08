"""
Main module to collect and log synchronized best bid/ask data from multiple exchanges.

This script launches one threaded WebSocket client per exchange (Binance, Bitmart, Bybit,
HashKey, HTX, Bitget, Deribit, Kuma, Hyperliquid). Each client publishes top-of-book
snapshots into a shared dict. A logger loop waits for updates and appends rows to a CSV.

Notes
-----
- All clients should set the provided 'threading.Event' when they update their part
  of 'DATA_GLOBAL'.
- Rows are only written when every exchange has populated all fields (no '-1' sentinels).
"""
import csv
import threading
from pathlib import Path
from typing import Dict, List

from binance_perptual_futures_orderbook import BINANCEOrderbookManager
from bitget_perptual_futures_orderbook import BITGETOrderbookManager
from bitmart_perptual_futures_orderbook import BITMARTOrderbookManager
from bybit_perptual_futures_orderbook import BYBITOrderbookManager
from deribit_perptual_futures_orderbook import DERIBITOrderbookManager
from htx_perpetual_futures_orderbook import HTXOrderbookManager
from hyperliquid_perptual_futures_orderbook import HYPERLIQUIDOrderbookManager
from kuma_perptual_futures_orderbook import KUMAOrderbookManager
from utils import get_timestamp_ms

SYMBOL = "sol"  # Base symbol (lowercase here; clients will format as needed)

EXCHANGES: List[str] = [
    "binance",
    "bitmart",
    "bybit",
    "htx",
    "bitget",
    "deribit",
    "kuma",
    "hyperliquid",
]

FIELDS: List[str] = ["timestamp", "bid_price", "ask_price", "bid_size", "ask_size", "latency"]

# Where to write data
FILE_PATH = Path("sol_spot_futures_data.csv")

# How many rows to buffer before writing to disk
BATCH_SIZE = 1_000_000

# Concurrency primitives
DATA_UPDATED = threading.Event()
LOCK = threading.Lock()

# Shared state: one dict per exchange
DATA_GLOBAL: Dict[str, Dict[str, float | int]] = {ex: {k: -1 for k in FIELDS} for ex in EXCHANGES}

def run_logger() -> None:
    """
    Collect updated rows from 'DATA_GLOBAL' and write them to CSV in batches.

    The function waits for 'DATA_UPDATED' to be set by any client, takes a
    snapshot across all exchanges, and if *every* field is populated (no '-1'),
    appends the row to an in-memory buffer. The buffer is flushed to disk when
    it reaches 'BATCH_SIZE.

    Returns
    -------
    None
    """
    rows: List[List[float | int]] = []
    while True:
        try:
            updated = DATA_UPDATED.wait(timeout=1)  # Block until data is updated or timeout
            DATA_UPDATED.clear()

            if not updated:
                continue

            machine_time = get_timestamp_ms()

            with LOCK:
                row: List[float | int] = []
                for ex in EXCHANGES:
                    row.append(machine_time)
                    ex_slice = DATA_GLOBAL[ex]
                    row.extend(ex_slice[fld] for fld in FIELDS)

            if all(v != -1 for v in row):
                rows.append(row)

            if len(rows) >= BATCH_SIZE:
                # Periodically write rows to file
                with FILE_PATH.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                rows.clear()

        except Exception as ex:
            # Avoid swallowing meaningful errors silently
            print(f"[logger] error: {ex}", flush=True)

if __name__ == "__main__":
    # Instantiate clients (one per exchange)
    # Each manager writes into DATA_GLOBAL[exchange] and sets DATA_UPDATED.
    mgr_binance = BINANCEOrderbookManager(
        url="wss://fstream.binance.com/ws",
        exchange=EXCHANGES[0],
        data=DATA_GLOBAL[EXCHANGES[0]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL}usdt",
        channel="bookTicker",
    )

    mgr_bitmart = BITMARTOrderbookManager(
        url="wss://openapi-ws-v2.bitmart.com/api?protocol=1.1",
        exchange=EXCHANGES[1],
        data=DATA_GLOBAL[EXCHANGES[1]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}USDT",
        channel="depthIncrease5",
        application_level_ping_interval_s=18,
        application_level_pong_timeout_s=18,
    )

    mgr_bybit = BYBITOrderbookManager(
        url="wss://stream.bybit.com/v5/public/linear",
        exchange=EXCHANGES[2],
        data=DATA_GLOBAL[EXCHANGES[2]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}USDT",
        channel="orderbook",
        application_level_ping_interval_s=20,
        application_level_pong_timeout_s=20,
    )

    mgr_htx = HTXOrderbookManager(
        url="wss://api.hbdm.com/linear-swap-ws",
        exchange=EXCHANGES[3],
        data=DATA_GLOBAL[EXCHANGES[3]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}-USDT",
        channel="bbo",
        application_level_ping_interval_s=30,
        application_level_pong_timeout_s=30,
    )

    mgr_bitget = BITGETOrderbookManager(
        url="wss://ws.bitget.com/v2/ws/public",
        exchange=EXCHANGES[4],
        data=DATA_GLOBAL[EXCHANGES[4]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}USDT",
        channel="books1",
        application_level_ping_interval_s=30,
        application_level_pong_timeout_s=30,
    )

    mgr_deribit = DERIBITOrderbookManager(
        url="wss://www.deribit.com/ws/api/v2",
        exchange=EXCHANGES[5],
        data=DATA_GLOBAL[EXCHANGES[5]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}_USDC-PERPETUAL",
        channel="book",
        application_level_pong_timeout_s=180,
    )

    mgr_kuma = KUMAOrderbookManager(
        url="wss://websocket.kuma.bid/v1",
        exchange=EXCHANGES[6],
        data=DATA_GLOBAL[EXCHANGES[6]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=f"{SYMBOL.upper()}-USD",
        channel="l2orderbook",
    )

    mgr_hl = HYPERLIQUIDOrderbookManager(
        url="wss://api.hyperliquid.xyz/ws",
        exchange=EXCHANGES[7],
        data=DATA_GLOBAL[EXCHANGES[7]],
        lock=LOCK,
        event=DATA_UPDATED,
        symbol=SYMBOL.upper(),
        channel="l2Book",
        application_level_ping_interval_s=50,
        application_level_pong_timeout_s=50,
    )

    # Start all clients
    for mgr in (
        mgr_binance,
        mgr_bitmart,
        mgr_bybit,
        mgr_htx,
        mgr_bitget,
        mgr_deribit,
        mgr_kuma,
        mgr_hl,
    ):
        mgr.start()

    # Start logger (blocks)
    run_logger()