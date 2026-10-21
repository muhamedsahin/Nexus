"""Isimli aktivasyon ve PythonModule govdelerini .nxs icine yazar.

Kapali hucre yalniz float, int veya bool olabilir. Yuklenen govde math ve
yerlesik adlari gorur. Baska bir modul adi yeni surecte yoktur.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import math
import marshal
import struct
import types

_STORE = {}
_MODULES = {}
_ACTIVATION = 1
_MODULE = 2


def _check(fn):
    if fn is None:
        return
    if not hasattr(fn, "__code__"):
        raise TypeError("bu cagri govdesi dosyaya yazilmaz")
    for cell in fn.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value < -9223372036854775808 or value > 9223372036854775807:
                raise TypeError("kapali degiskenli fonksiyon dosyaya yazilmaz")
            continue
        if isinstance(value, float):
            continue
        raise TypeError("kapali degiskenli fonksiyon dosyaya yazilmaz")


def remember(name, forward, backward=None):
    _check(forward)
    _check(backward)
    _STORE[name] = (forward, backward)


def remember_module(name, forward, backward=None, shapes=None):
    _check(forward)
    _check(backward)
    _MODULES[name] = (forward, backward, [list(shape) for shape in (shapes or [])])


def _write_code(buf, fn):
    blob = b"" if fn is None else marshal.dumps(fn.__code__)
    buf += struct.pack("<I", len(blob))
    buf += blob
    cells = () if fn is None or not fn.__closure__ else fn.__closure__
    buf += struct.pack("<I", len(cells))
    for cell in cells:
        value = cell.cell_contents
        if isinstance(value, bool):
            buf += struct.pack("<Bq", 3, int(value))
        elif isinstance(value, int):
            buf += struct.pack("<Bq", 2, value)
        else:
            buf += struct.pack("<Bd", 1, float(value))


def _write_record(buf, kind, name, forward, backward, shapes):
    name_bytes = name.encode("utf-8")
    buf += struct.pack("<B", kind)
    buf += struct.pack("<I", len(name_bytes)) + name_bytes
    _write_code(buf, forward)
    _write_code(buf, backward)
    if kind == _MODULE:
        buf += struct.pack("<I", len(shapes))
        for shape in shapes:
            buf += struct.pack("<I", len(shape))
            for dim in shape:
                buf += struct.pack("<Q", int(dim))


def pack():
    if not _STORE and not _MODULES:
        return b""
    buf = bytearray(b"NXPY")
    records = [( _ACTIVATION, name, forward, backward, []) for name, (forward, backward) in _STORE.items()]
    records.extend((_MODULE, name, forward, backward, shapes) for name, (forward, backward, shapes) in _MODULES.items())
    buf += struct.pack("<HI", 1, len(records))
    for record in records:
        _write_record(buf, *record)
    return bytes(buf)


def _cell(value):
    return (lambda: value).__closure__[0]


def _load_fn(blob, cells):
    code = marshal.loads(blob)
    closure = tuple(_cell(value) for value in cells) if cells else None
    return types.FunctionType(code, {"__builtins__": __builtins__, "math": math}, closure=closure)


def _read_code(view, offset):
    (length,) = struct.unpack_from("<I", view, offset)
    offset += 4
    blob = b""
    if length:
        blob = bytes(view[offset : offset + length])
        offset += length
    (cell_count,) = struct.unpack_from("<I", view, offset)
    offset += 4
    cells = []
    for _ in range(cell_count):
        (kind, raw) = struct.unpack_from("<Bd", view, offset)
        offset += 9
        if kind == 2:
            (raw,) = struct.unpack_from("<q", view, offset - 8)
            cells.append(int(raw))
        elif kind == 3:
            (raw,) = struct.unpack_from("<q", view, offset - 8)
            cells.append(bool(raw))
        else:
            cells.append(float(raw))
    fn = _load_fn(blob, cells) if blob else None
    return fn, offset


def _read_code_only(view, offset):
    (length,) = struct.unpack_from("<I", view, offset)
    offset += 4
    fn = None
    if length:
        fn = _load_fn(bytes(view[offset : offset + length]), [])
        offset += length
    return fn, offset


def _restore_legacy(view):
    from nexus._core.nn import activation as raw

    (count,) = struct.unpack_from("<I", view, 0)
    offset = 4
    for _ in range(count):
        (name_len,) = struct.unpack_from("<I", view, offset)
        offset += 4
        name = bytes(view[offset : offset + name_len]).decode("utf-8")
        offset += name_len
        forward, offset = _read_code_only(view, offset)
        backward, offset = _read_code_only(view, offset)
        remember(name, forward, backward)
        raw(name, forward) if backward is None else raw(name, forward, backward)


def restore(blob):
    if not blob:
        return
    view = memoryview(blob)
    if bytes(view[:4]) != b"NXPY":
        _restore_legacy(view)
        return
    from nexus._core.nn import PythonModule as raw_module
    from nexus._core.nn import activation as raw_activation

    (version, count) = struct.unpack_from("<HI", view, 4)
    if version != 1:
        raise ValueError("nxs python payload is unsupported")
    offset = 10
    for _ in range(count):
        (kind,) = struct.unpack_from("<B", view, offset)
        offset += 1
        (name_len,) = struct.unpack_from("<I", view, offset)
        offset += 4
        name = bytes(view[offset : offset + name_len]).decode("utf-8")
        offset += name_len
        forward, offset = _read_code(view, offset)
        backward, offset = _read_code(view, offset)
        shapes = []
        if kind == _MODULE:
            (n_param,) = struct.unpack_from("<I", view, offset)
            offset += 4
            for _shape in range(n_param):
                (rank,) = struct.unpack_from("<I", view, offset)
                offset += 4
                dims = []
                for _dim in range(rank):
                    (dim,) = struct.unpack_from("<Q", view, offset)
                    offset += 8
                    dims.append(dim)
                shapes.append(dims)
            remember_module(name, forward, backward, shapes)
            raw_module(name, forward, backward, shapes)
        else:
            remember(name, forward, backward)
            raw_activation(name, forward) if backward is None else raw_activation(name, forward, backward)
