import json
import threading

import websocket # websocket-client

from client_deribit import ClientDERIBIT
from utils import get_timestamp_ms, get_monotonic_s

class DERIBITOrderbookManager(ClientDERIBIT):
    """
    Deribit USDT-Futures L1 order book manager.

    Parameters
    ----------
    url : str
        WebSocket URL.
    exchange : str
        Exchange name (for logs/metrics).
    symbol : str
        Instrument ID (e.g., "BTC-PERPETUAL", "SOL_USDC-PERPETUAL).
    data : dict
        Shared dict to write top-of-book fields into:
        {'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'latency'}.
    lock : threading.Lock
        Lock protecting shared 'data'.
    event : threading.Event
        Event set when a fresh top-of-book tick is written.
    channel : str
        Deribit channel name, e.g. "book.<instrument_name>.<group>.<depth>.<interval>".
    application_level_pong_timeout_s : float
        Max time (seconds) allowed since last pong before closing the socket.

    Notes
    -----
    - Protocol-level pings ('ping_interval', 'ping_timeout') are orthogonal and handled
      by the base client. This class manages Deribit's app-level pings.
    """
    # Call init from parent class
    def __init__(self, url: str, exchange: str, data: dict[str, float | int], lock: threading.Lock, event: threading.Event, symbol: str, channel: str, application_level_pong_timeout_s: int) -> None:
        super().__init__(url, exchange)
        # References & sync
        self.data: dict[str] = data
        self._lock: threading.Lock = lock
        self.event: threading.Event = event

        # Identity/config
        self._symbol: str = symbol
        self._channel: str = channel
        self._pong_timeout_s: float = application_level_pong_timeout_s

        # Last update id
        self._lastUpdateId: int = 0

        # Heartbeat state
        self._hb_stop: threading.Event = threading.Event()
        self._last_pong_s: float = get_monotonic_s()

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
                params = msg.get('params', {})
                msg_type = params.get('type')
                msg_channel = params.get('channel', [])

                if msg_type == 'heartbeat':
                    self._last_pong_s = get_monotonic_s()

                elif msg_type == 'test_request':
                    self._last_pong_s = get_monotonic_s()
                    ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": 42,
                        "method": "/api/v2/public/test",
                        "params": {}
                    }
                    ))

                elif self._channel in msg_channel:
                    data = params.get('data', {})
                    ts = data.get('timestamp', 0)
                    bid_price = data.get('bids')[0][0]
                    ask_price = data.get('asks')[0][0]
                    bid_size = data.get('bids')[0][1]
                    ask_size = data.get('asks')[0][1]

                    self.data['bid_size'] = bid_size
                    self.data['ask_size'] = ask_size

                    if self.data['bid_price'] != bid_price or self.data['ask_price'] != ask_price:
                        with self._lock:
                            self.data['timestamp'] = ts
                            self.data['bid_price'] = bid_price
                            self.data['ask_price'] = ask_price
                            self.data['latency'] = max(get_timestamp_ms() - ts, 0)
                        self.event.set()
                else:
                    # Deribit sends subscription confirmation messages
                    return
                
            else:
                print(f"[{self._exchange}] unknown message: {message}", flush = True)

        except Exception as ex:
            print(f"[{self._exchange}] error in on_message: {ex}, {msg}", flush=True)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Subscribe to the book channel and start watchdog.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        sub_msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "id": 42,
            "params": {
                "channels": [f"{self._channel}.{self._symbol}.none.1.100ms"]
            }
        }

        heartbeat_msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "public/set_heartbeat",
            "params": {
                "interval": 20
            }
        }

        ws.send(json.dumps(heartbeat_msg))
        ws.send(json.dumps(sub_msg))

        # Init heartbeat state and start heartbeat threads
        self._last_pong_s = get_monotonic_s()
        self._hb_stop.clear()
        threading.Thread(target=self._watchdog_loop, args=(ws,), daemon=True).start()

    def on_close(self, ws: websocket.WebSocketApp, code: int | None, reason: str | None) -> None:
        """
        Stop heartbeat threads on close.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            WebSocket connection (closing/closed).
        code : int or None
            Close code.s
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

