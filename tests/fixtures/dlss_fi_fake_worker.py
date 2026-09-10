"""CPU-only malformed/fragmented pipe peer; not a DLSS implementation."""
import os
import struct
import sys
import time

mode = sys.argv[1]


def read(count):
    out = bytearray()
    while len(out) < count:
        chunk = sys.stdin.buffer.read(count-len(out))
        if not chunk:
            raise SystemExit(0)
        out.extend(chunk)
    return bytes(out)


def emit(data):
    if mode == "fragment":
        for byte in data:
            os.write(1, bytes((byte,)))
    else:
        os.write(1, data)


if mode == "hang_setup":
    time.sleep(60)
_, width, height, count, generated = struct.unpack("<5I", read(20))
if mode == "stderr_flood":
    for _ in range(128):
        os.write(2, b"x"*4096)
emit(struct.pack("<4I", 0 if mode == "bad_setup" else 0x31524746, 0, 1, 0))
for index in range(count):
    _, _, reset, _, _, _ = struct.unpack("<4I2q", read(32))
    if mode == "hang_input":
        time.sleep(60)
    read(width*height*8)
    if mode == "hang_frame":
        time.sleep(60)
    n = 1
    if mode == "no_frame" and not reset:
        n = 0
    if mode == "huge_count":
        n = 0xFFFFFFFF
    emit(struct.pack("<4I", 0x314F4746, 0, n, int(mode == "disabled")))
    if mode == "truncated":
        emit(b"a")
        raise SystemExit(0)
    if n == 1:
        emit(bytes([index+1])*width*height*4)
time.sleep(60)
