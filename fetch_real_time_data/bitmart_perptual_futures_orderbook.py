import json
import threading

import websocket # websocket-client

from client_bitmart import ClientBITMART
from utils import get_timestamp_ms, get_monotonic_s

class BITMARTOrderbookManager(ClientBITMART):
    """
    Bitmart USDT-Futures L2 order book manager with application-level heartbeats.

    Parameters
    ----------
    url : str
        WebSocket URL.
    exchange : str
        Exchange name (for logs/metrics).
    symbol : str
        Instrument ID (e.g., "BTCUSDT").
    data : dict
        Shared dict to write top-of-book fields into:
        {'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'latency'}.
    lock : threading.Lock
        Lock protecting shared 'data'.
    event : threading.Event
        Event set when a fresh top-of-book tick is written.
    channel : str
        Bitmart channel name, e.g. "depthIncrease5:<symbol>@<speed>".
    application_level_ping_interval_s : float
        Interval (seconds) to send application-level '"ping"' (Bitmart expects a certain ping frame).
    application_level_pong_timeout_s : float
        Max time (seconds) allowed since last pong before closing the socket.

    Notes
    -----
    - Bitmart replies with a pong frame.
    - Protocol-level pings ('ping_interval', 'ping_timeout') are orthogonal and handled
      by the base client. This class manages Bitmart's app-level ping/pong.
    """
    # Call init from parent class
    def __init__(self, url: str, exchange: str, data: dict[str, float | int], lock: threading.Lock, event: threading.Event, symbol: str, channel: str, application_level_ping_interval_s: int, application_level_pong_timeout_s: int) -> None:
        super().__init__(url, exchange)
        # References & sync
        self.data: dict[str] = data
        self._lock: threading.Lock = lock
        self.event: threading.Event = event

        # Identity/config
        self._symbol: str = symbol
        self._channel: str = channel
        self._ping_interval_s: float = application_level_ping_interval_s
        self._pong_timeout_s: float = application_level_pong_timeout_s

        # Heartbeat state
        self._hb_stop: threading.Event = threading.Event()
        self._last_pong_s: float = get_monotonic_s()

        # Sequence number and orderbook
        self._sequence_number = -1
        self._orderbook = {'bids': {}, 'asks': {}}

    def _ping_loop(self, ws: websocket.WebSocketApp) -> None:
        """
        Send Bitmart application-level 'ping' or ping frame at a fixed cadence.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        while not self._hb_stop.is_set() and ws.sock and ws.sock.connected:
            try:
                ws.send(json.dumps({"action":"ping"})) # Bitmart expects this ping frame
            except Exception as ex:
                print(f"[{self._exchange}] _ping_loop error: {ex}", flush = True)
                break

            if self._hb_stop.wait(self._ping_interval_s):
                break

    def _watchdog_loop(self, ws: websocket.WebSocketApp) -> None:
        """
        Close the socket if a 'pong' hasn't been seen within the timeout.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        while not self._hb_stop.is_set() and ws.sock and ws.sock.connected:
            if get_monotonic_s() - self._last_pong_s > self._pong_timeout_s:
                try:
                    print("pong timeout")
                    ws.close()
                finally:
                    break
            self._hb_stop.wait(1)

    def _update_orderbook(self, bids: dict[str, str] | dict, asks: dict[str, str] | dict) -> None:
        """
        Apply incremental updates to the local order book state.

        Each element in 'bids or 'asks is expected to be a dict with
        the keys 'price' and 'vol' (both strings).

        Rules
        -----
        - If 'vol' == 0, the price level is removed.
        - Otherwise, the price level is inserted or updated.
        - If 'bids' or 'asks' is an empty dict, no changes are applied on that side.

        Parameters
        ----------
        bids : dict
            List of bid updates. Each dict should have fields:
            - 'price' : str
            - 'vol' : str
        asks : dict
            List of ask updates. Each dict should have fields:
            - 'price' : str
            - 'vol' : str

        Returns
        -------
        None

        Notes
        -----
        - Price levels are stored as 'float(price) -> float(size)' mappings.
        - This does not perform sequence number validation; caller is
        responsible for ensuring updates are applied in the right order.
        """
        # Update/initialize bids
        for elem in bids:
            price = elem.get('price')
            size = elem.get('vol')
            if float(size) == 0:
                self._orderbook['bids'].pop(float(price), None)
            else:
                self._orderbook['bids'][float(price)] = float(size)
        # Update/initialize asks
        for elem in asks:
            price = elem.get('price')
            size = elem.get('vol')
            if float(size) == 0:
                self._orderbook['asks'].pop(float(price), None)
            else:
                self._orderbook['asks'][float(price)] = float(size)

    def on_message(self, ws: websocket.WebSocketApp, message: bytes) -> None:
        """
        Handles incoming WebSocket messages and L2 data and updates the book of top 5 levels of each side.

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

                if "pong" in msg.get('data', {}):
                    self._last_pong_s = get_monotonic_s()

                elif msg.get('action') == 'subscribe':
                    # Bitmart sends subscription confirmation messages
                    return

                elif msg.get('data').get('type') == 'snapshot':
                    # Initialize sequence number
                    self._sequence_number = msg.get('data').get('version')
                    ts = msg.get('data').get('ms_t')
                    # Initialize orderbook
                    self._orderbook = {'bids': {}, 'asks': {}}
                    data_bids = msg.get('data').get('bids')
                    data_asks = msg.get('data').get('asks')
                    self._update_orderbook(data_bids, data_asks)

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

                elif msg.get('data').get('type') == 'update':
                    if msg.get('data').get('version') == self._sequence_number + 1:
                        self._sequence_number += 1                
                        # Ignore the next 5 lines
                        ts = msg.get('data').get('ms_t')
                        bid_price = self.data['bid_price']
                        ask_price = self.data['ask_price']
                        bid_size = self.data['bid_size']
                        ask_size = self.data['ask_size']

                        data_bids = msg.get('data').get('bids')
                        data_asks = msg.get('data').get('asks')
                        self._update_orderbook(data_bids, data_asks)

                        bid_price, bid_size = max(self._orderbook['bids'].items(), default=(None, None))  
                        ask_price, ask_size = min(self._orderbook['asks'].items(), default=(None, None))  

                        self.data['bid_size'] = bid_size
                        self.data['ask_size'] = ask_size

                        if self.data['bid_price'] != bid_price or self.data['ask_price'] != ask_price:
                            with self._lock:
                                self.data['timestamp'] = ts
                                self.data['bid_price'] = bid_price
                                self.data['ask_price'] = ask_price
                                self.data['latency'] = max(get_timestamp_ms() - ts, 0)
                            self.event.set()
                    
                    elif msg.get('data').get('version') > self._sequence_number + 1:
                        ws.close()

                    else:
                        return

                else:
                    print(f"[{self._exchange}] unknown message: {msg}", flush = True)

            else:
                print(f"[{self._exchange}] unknown message: {message}", flush = True)

        except Exception as ex:
            print(f"[{self._exchange}] error in on_message: {ex}, {msg}", flush = True)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Subscribe to the depthIncrease5 channel and start heartbeats.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        sub_msg = {
            "action": "subscribe",
            "args": [f"futures/{self._channel}:{self._symbol}@100ms"]
        }
        ws.send(json.dumps(sub_msg))

        # Init heartbeat state and start heartbeat threads
        self._last_pong_s = get_monotonic_s()
        self._hb_stop.clear()
        threading.Thread(target=self._ping_loop, args=(ws,), daemon=True).start()
        threading.Thread(target=self._watchdog_loop, args=(ws,), daemon=True).start()

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
