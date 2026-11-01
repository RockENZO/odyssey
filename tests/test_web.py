"""Bound remote reads and keep the visible text limits honest."""
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from odyssey.tools import web


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/error':
            self.send_error(500, 'boom')
            return
        if self.path == '/big':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            chunk = b'<p>' + b'x' * 65_530 + b'</p>'
            try:
                for _ in range(2_000):  # 128 MB if the client reads everything
                    self.wfile.write(chunk)
                    self.server.bytes_written += len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        body = b'<html><head><style>p{color:red}</style></head><body><script>hidden()</script><p>Hello <b>world</b></p></body></html>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.bytes_written = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def url_for(server, path='/'):
    return f'http://127.0.0.1:{server.server_port}{path}'


def test_read_url_extracts_visible_text(site):
    result = asyncio.run(web.read_url(url_for(site)))
    assert result == 'Hello world'


def test_read_url_stops_after_the_byte_budget(site):
    result = asyncio.run(web.read_url(url_for(site, '/big')))
    assert result.endswith('... [truncated]')
    assert len(result) == web.MAX_URL_TEXT_CHARS + len('... [truncated]')
    # The server would send 128 MB; a bounded client must disconnect early.
    assert site.bytes_written < 8_000_000


def test_read_url_reports_http_errors(site):
    result = asyncio.run(web.read_url(url_for(site, '/error')))
    assert result.startswith('Failed to read URL:')
