import json
import threading

import websocket # websocket-client

from client_binance import ClientBINANCE
from utils import get_timestamp_ms

class BINANCEOrderbookManager(ClientBINANCE):
    """
    Binance USDT-Futures L1 order book manager.

    Parameters
    ----------
    url : str
        WebSocket URL.
    exchange : str
        Exchange name (for logs/metrics).
    symbol : str
        Instrument ID (e.g., "btcusdt").
    data : dict
        Shared dict to write top-of-book fields into:
        {'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'latency'}.
    lock : threading.Lock
        Lock protecting shared 'data'.
    event : threading.Event
        Event set when a fresh top-of-book tick is written.
    channel : str
        Binance channel name, e.g. "bookTicker:<symbol>".
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

        # Last update id
        self._lastUpdateId: int = 0

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
                if message == b"ping":
                    ws.send("pong", opcode=websocket.ABNF.OPCODE_PONG)
                
                else:
                    msg = json.loads(message)
                    if msg == {'result': None, 'id': 1}:
                        # Binance sends subscription confirmation messages
                        return

                    elif msg.get('u') > self._lastUpdateId:
                        self._lastUpdateId = msg['u']
                        ts = msg['T']

                        bid_price = float(msg['b'])
                        ask_price = float(msg['a'])
                        bid_size = float(msg['B'])
                        ask_size = float(msg['A'])

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
                        print(f"[{self._exchange}] unknown message: {msg}", flush = True)

            else:
                print(f"[{self._exchange}] unknown message: {message}", flush = True)

        except Exception as ex:
            print(f"[{self._exchange}] error in on_message: {ex}, {msg}", flush = True)


    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Subscribe to the bookTicker channel.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            Active WebSocket connection.
        """
        sub_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{self._symbol}@{self._channel}"],
            "id": 1
        }
        ws.send(json.dumps(sub_msg))
