import json
import threading

import websocket # websocket-client

from client_hyperliquid import ClientHYPERLIQUID
from utils import get_timestamp_ms, get_monotonic_s

class HYPERLIQUIDOrderbookManager(ClientHYPERLIQUID):
    """
    Hyperliquid USDT-Futures L1 order book manager.

    Parameters
    ----------
    url : str
        WebSocket URL.
    exchange : str
        Exchange name (for logs/metrics).
    symbol : str
        Instrument ID (e.g., "BTC").
    data : dict
        Shared dict to write top-of-book fields into:
        {'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'latency'}.
    lock : threading.Lock
        Lock protecting shared 'data'.
    event : threading.Event
        Event set when a fresh top-of-book tick is written.
    channel : str
        Hyperliquid channel name, e.g. "l2Book".
    application_level_ping_interval_s : float
        Interval (seconds) to send application-level '"ping"' (Hyperliquid expects a certain ping frame).
    application_level_pong_timeout_s : float
        Max time (seconds) allowed since last pong before closing the socket.

    Notes
    -----
    - Hyperliquid replies with a pong frame.
    - Protocol-level pings ('ping_interval', 'ping_timeout') are orthogonal and handled
      by the base client. This class manages Hyperliquid's app-level ping/pong.
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

    def _ping_loop(self, ws: websocket.WebSocketApp) -> None:
        """
        Send Hyperliquid application-level 'ping' or ping frame at a fixed cadence.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        while not self._hb_stop.is_set() and ws.sock and ws.sock.connected:
            try:
                ws.send(json.dumps({"method": "ping"})) # Hyperliquid expects this ping frame
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

    def on_message(self, ws: websocket.WebSocketApp, message: bytes) -> None:
        """
        Handles incoming WebSocket messages and updates L1 data.

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
                msg_channel = msg.get('channel')
                if msg_channel == self._channel:
                    ts = float(msg.get('data').get('time'))
                    bid_price = float(msg['data']['levels'][0][0]['px'])
                    bid_size = float(msg['data']['levels'][0][0]['sz'])
                    ask_price = float(msg['data']['levels'][1][0]['px'])
                    ask_size = float(msg['data']['levels'][1][0]['sz'])

                    self.data['bid_size'] = bid_size
                    self.data['ask_size'] = ask_size

                    if self.data['bid_price'] != bid_price or self.data['ask_price'] != ask_price:
                        with self._lock:
                            self.data['timestamp'] = ts
                            self.data['bid_price'] = bid_price
                            self.data['ask_price'] = ask_price
                            self.data['latency'] = get_timestamp_ms() - ts
                        self.event.set()

                elif msg_channel == 'pong':
                    self._last_pong_s = get_monotonic_s()

                else:
                    # Hyperliquid sends subscription confirmation messages
                    return
                
            else:
                print(f"[{self._exchange}] unknown message: {message}", flush = True)

        except Exception as ex:
            print(f"[{self._exchange}] error in on_message: {ex}, {msg}", flush = True)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Subscribe to the orderbook channel and start heartbeats.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        sub_msg = {
            "method": "subscribe",
            "subscription": {
                "type": self._channel,
                "coin": self._symbol
            }
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
