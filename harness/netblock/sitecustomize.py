"""запрещает тестам выход в сеть, как в песочнице, питон подхватывает файл сам через PYTHONPATH"""

import socket

_LOCAL = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _is_local(address) -> bool:
    if not isinstance(address, tuple):  # unix сокеты и прочее локальное
        return True
    host = address[0]
    return not isinstance(host, str) or host in _LOCAL or host.startswith("127.")


def _guarded_connect(self, address):
    if not _is_local(address):
        raise OSError(101, "Network is unreachable (no internet in the sandbox)")
    return _connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_local(address):
        return 101
    return _connect_ex(self, address)


_getaddrinfo = socket.getaddrinfo


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_local((host,)):
        raise socket.gaierror(-3, "Temporary failure in name resolution (no internet in the sandbox)")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.getaddrinfo = _guarded_getaddrinfo
