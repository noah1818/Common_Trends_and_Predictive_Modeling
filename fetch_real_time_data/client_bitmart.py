import socket
import threading
import time

import websocket

class ClientBITMART(threading.Thread):
    """
    WebSocket client wrapper for maintaining a persistent connection
    with automatic reconnection and ping/pong handling.

    This client runs in a separate thread, connects to a WebSocket server,
    and handles messages, errors, and reconnections.

    Parameters
    ----------
    url : str
        The WebSocket server URL.
    exchange : str
        Identifier for the exchange or data source (used for logging).
    
    Notes
    -----
    This module defines the base 'ClientBITMART' class, which manages persistent
    WebSocket connection mechanisms (including reconnection logic, ping/pong handling, and
    error recovery).

    Specialized client implementations can extend this class to process specific
    data streams. For example:

    - 'bitmart_perptual_futures_orderbook.py' implements 'BITMARTOrderbookManager', a WebSocket client for
    handling L1 orderbook data in real-time.

    Both this base client and its WebSocket subclasses are implemented according to 
    the official Bitmart Exchange API documentation as of 2025-08-25. 
    Reference: https://developer-pro.bitmart.com

    The 'on_open' and 'on_message' methods are intentionally public to serve as callback hooks
    for 'websocket.WebSocketApp', they are expected to override these
    to implement custom behavior.
    """
    def __init__(self, url: str, exchange: str) -> None:
        super().__init__()
        self._url = url
        self._exchange = exchange
        # Disable Nagle's algorithm (TCP_NODELAY = 1) to reduce latency
        # This ensures small messages are sent immediately rather than buffered
        self._sockopt = ((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)
        self._ws = None
        self._ping_interval = 40  # Seconds to send a ping
        self._ping_timeout = 20  # Seconds to wait for a pong after sending a ping
        self._sleep_reconnect = 2

    def run(self) -> None:
        """
        Main loop for establishing and maintaining the WebSocket connection.

        A new 'WebSocketApp' instance is created for each connection attempt.
        If the connection drops or fails, the client waits for a short period
        before retrying.

        Returns
        -------
        None
        """
        while True:
            self.ws = websocket.WebSocketApp(
                url=self._url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            try:
                self.ws.run_forever(
                    ping_interval=self._ping_interval, 
                    ping_timeout=self._ping_timeout, 
                    sockopt = self._sockopt, 
                    skip_utf8_validation = True # Improves performance; safe because exchange guarantees valid UTF-8
                )

            except Exception as ex:
                print(f"{self._exchange} run_forever exception: {ex}", flush=True)

            time.sleep(self._sleep_reconnect)
    
    # Public hooks (override in subclasses)
    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Called when the WebSocket connection is successfully opened.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        
        Returns
        -------
        None
        """
        pass

    def on_message(self, ws: websocket.WebSocketApp, message: bytes) -> None:
        """
        Called when a new message is received.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        message : bytes
            The raw message received, typically just bytes.

        Returns
        -------
        None
        """
        pass

    def on_close(self, ws: websocket.WebSocketApp, code: int | None, reason: str | None) -> None: 
        """
        Called when the WebSocket connection is closed.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        code : int or None
            The status code for the closure.
        reason : str or None
            The reason for closure provided by the server.

        Returns
        -------
        None
        """
        pass

    def on_error(self, ws: websocket.WebSocketApp, error: Exception | str | None) -> None:
        """
        Called when an error occurs during WebSocket communication.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        error : Exception or str or None
            The exception that occurred.

        Returns
        -------
        None
        """
        pass

    # Internal wrappers wired to WebSocketApp
    def _on_close(self, ws: websocket.WebSocketApp, code: int | None, reason: str | None) -> None:
        """
        Internal: dispatch close to subclass hook and log.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        code : int or None
            The status code for the closure.
        reason : str or None
            The reason for closure provided by the server.
        
        Returns
        -------
        None
        """
        try:
            self.on_close(ws, code, reason)
        except Exception as ex:
            print(f"[{self._exchange}] on_close hook error: {ex}", flush=True)

        print(f"[{self._exchange}] WebSocket closed (code={code}, reason={reason})", flush=True)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception | str | None) -> None:
        """
        Internal: dispatch error to subclass hook, then close the socket.

        Parameters
        ----------
        ws : websocket.WebSocketApp
            The WebSocketApp instance managing the connection.
        error : Exceptionor str or None
            The exception that occurred.
        
        Returns
        -------
        None
        """
        try:
            self.on_error(ws, error)
        except Exception as ex:
            print(f"[{self._exchange}] on_error hook error: {ex}", flush=True)
        
        print(f"[{self._exchange} on_error: {error}]", flush = True)
        try:
            ws.close()
        except Exception as ex:
            print(f"[{self._exchange}] on_error close exception: {ex}", flush=True)