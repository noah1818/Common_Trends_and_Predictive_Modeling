import json
import threading

import websocket # websocket-client
import requests # requests
import pandas as pd # pandas

from client_kuma import ClientKUMA
from utils import get_timestamp_ms, get_monotonic_s

class KUMAOrderbookManager(ClientKUMA):
    """
    Kuma USDT-Futures L1 order book manager.

    Parameters
    ----------
    url : str
        WebSocket URL.
    exchange : str
        Exchange name (for logs/metrics).
    symbol : str
        Instrument ID (e.g., "BTC-USD").
    data : dict
        Shared dict to write top-of-book fields into:
        {'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'latency'}.
    lock : threading.Lock
        Lock protecting shared 'data'.
    event : threading.Event
        Event set when a fresh top-of-book tick is written.
    channel : str
        Kuma channel name, e.g. "l2orderbook".
    """
    # Call init from parent class
    def __init__(self, url: str, exchange: str, data: dict[str, float | int], lock: threading.Lock, event: threading.Event, symbol: str, channel: str) -> None:
        super().__init__(url, exchange)
        # References & sync
        self.data: dict[str] = data
        self._lock: threading.Lock = lock
        self.event: threading.Event = event

        # Identity/config
        self._symbol: str = symbol
        self._channel: str = channel

        # Heartbeat state
        self._hb_stop: threading.Event = threading.Event()
        self._last_pong_s: float = get_monotonic_s()

        # Sequence number and orderbook
        self._sequence_number = -1
        self._orderbook = {'bids': {}, 'asks': {}, 'lastUpdateId': 0}

    def update_orderbook(self, bids: list[list[str, str, str]] | list, asks: list[list[str, str, str]] | list) -> None:
        """
        Apply incremental updates to the local order book state.

        Each element in 'bids or 'asks is expected to be a list with
        the keys 'price' and 'size' (both strings).

        Rules
        -----
        - If 'size' == 0, the price level is removed.
        - Otherwise, the price level is inserted or updated.
        - If 'bids' or 'asks' is an empty list, no changes are applied on that side.

        Parameters
        ----------
        bids : list
            List of bid updates. Each list should have fields:
            - 'price' : str
            - 'size' : str
            - 'orders': str
        asks : list
            List of ask updates. Each list should have fields:
            - 'price' : str
            - 'size' : str
            - 'orders': str

        Returns
        -------
        None

        Notes
        -----
        - Price levels are stored as 'float(price) -> float(size)' mappings.
        - We will ignore the number of orders at each price level
        - This does not perform sequence number validation; caller is
        responsible for ensuring updates are applied in the right order.
        """
        for price, size, _ in bids:
            if float(size) == 0:
                self._orderbook['bids'].pop(float(price), None)
            else:
                self._orderbook['bids'][float(price)] = float(size)

        for price, size, _ in asks:
            if float(size) == 0:
                self._orderbook['asks'].pop(float(price), None)
            else:
                self._orderbook['asks'][float(price)] = float(size)

    def on_message(self, ws: websocket.WebSocketApp, message: bytes) -> None:
        """
        Handles incoming WebSocket messages, L2 data and updates the order book.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        message : bytes
            Incoming frame payload.
        """
        try:
            if isinstance(message, bytes):
                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == self._channel:
                    current_update_id = msg.get('data').get('u')
                    ts = msg.get('data').get('t')

                    if current_update_id <= self._sequence_number:
                        # Discard all order book update messages with sequence numbers ess than or equal to the snapshot sequence number.
                        return

                    elif current_update_id == self._sequence_number + 1:
                        updated_bids = msg.get('data').get('b')
                        updated_asks = msg.get('data').get('a')

                        # Update current orderbook id
                        self._sequence_number += 1

                        self.update_orderbook(updated_bids, updated_asks)

                        # Get the TOB for both sides
                        bid_price, bid_size = max(self._orderbook['bids'].items(), default=(None, None))  
                        ask_price, ask_size = min(self._orderbook['asks'].items(), default=(None, None))

                        self.data['bid_size'] = bid_size
                        self.data['ask_size'] = ask_size

                        if self.data['bid_price'] != bid_price or self.data['ask_price'] != ask_price:
                            with self._lock:
                                self.data['timestamp'] = ts
                                self.data['bid_price'] = bid_price
                                self.data['ask_price'] = ask_price
                                self.data['latency'] = get_timestamp_ms() - ts
                            self.event.set()

                    else:
                        print("out of sync, re-syncing ...", flush=True)
                        self.populate_orderbook()

                elif msg_type == 'ping':
                    self._last_pong_s = get_monotonic_s()

                else:
                    return
                
            else:
                print(f"[{self._exchange}] unknown message: {message}", flush = True)

        except Exception as ex:
            print(f"[{self._exchange}] error in on_message: {ex}, {msg}", flush = True)

    def populate_orderbook(self) -> None:
        self._orderbook = {'bids': {}, 'asks': {}, 'lastUpdateId': 0}
        base = "https://api.kuma.bid/"
        endpoint = "/v1/orderbook"
        params = {
            "market": self._symbol,
            "level": 2,
            "limit": 50
        }
        r = requests.get(base + endpoint, params=params)
        data = r.json()

        data_bids = data['bids']
        data_asks = data['asks']

        self.update_orderbook(data_bids, data_asks)
        self._sequence_number = data['sequence']
        print(f"populated orderbook now, @ {pd.to_datetime(get_timestamp_ms(), unit='ms')}", flush=True)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Subscribe to the orderbook channel.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        sub_msg = {
            "method": "subscribe",
            "subscriptions": [
                {
                    "name": self._channel,
                    "markets": [self._symbol]
                }
            ]
        }
        ws.send(json.dumps(sub_msg))

    def on_close(self, ws: websocket.WebSocketApp, code: int | None, reason: str | None) -> None:
        """
        Stop heartbeat threads on close.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            WebSocket connection (closing/closed).
        code : int or None
            Close code.
        reason : str or None
            Close reason.
        """
        self._hb_stop.set()

    def on_error(self, ws: websocket.WebSocketApp, error: Exception | str | None) -> None:
        """
        Stop heartbeat threads when a WebSocket error occurs.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        error : Exception or str or None
            Error object or message provided by websocket-client.
        """
        self._hb_stop.set()

