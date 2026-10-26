# -*- coding: utf-8 -*-
"""live.py -- the gate, pointed at a running game instead of a mirror I wrote.

Up to here every claim in this repo was checked against a world this repo also
built, which is the weakest possible arrangement: the same head that wrote the
plant wrote the gate's picture of the world. This module removes that. The gate
still computes with embodied.py, unchanged; what changes is where `blocks` comes
from -- a live Minecraft server, through the body's own chunk cache, one column at
a time.

The line is the same one as before, only longer:

    bench (scripted or a model's plan)  ->  EmbodiedGate  ->  body  ->  server
                    ^                          |                        |
                    +------  its world  <------+---  chunk cache  <-----+

Three pieces:

    Rcon        set up a scene and place the body (the bench's job, not the gate's)
    LineLink    framed JSON over a socket, one line at a time, both directions
    RemoteWorld a VoxelWorld-shaped `blocks` mapping backed by column queries

RemoteWorld is deliberately a drop-in for VoxelWorld.blocks: the gate does
world.blocks.get((x, y, z)) and nothing else, so pointing that at a socket is the
whole change -- the gate's arithmetic, its five rules and its refusal messages are
the same code that was checked against the simulator.
"""
import json
import math
import socket
import struct


class Rcon:
    """Minecraft's RCON protocol: length, id, type, body, two nulls."""

    AUTH, CMD = 3, 2

    def __init__(self, host='127.0.0.1', port=25575, password='spine', timeout=30):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._send(self.AUTH, password)
        self._recv()

    def _send(self, kind, body):
        payload = struct.pack('<ii', 0, kind) + body.encode('utf-8') + b'\x00\x00'
        self.sock.sendall(struct.pack('<i', len(payload)) + payload)

    def _recv(self):
        raw = self.sock.recv(4)
        if len(raw) < 4:
            raise ConnectionError('rcon closed')
        need = struct.unpack('<i', raw)[0]
        buf = b''
        while len(buf) < need:
            chunk = self.sock.recv(need - len(buf))
            if not chunk:
                raise ConnectionError('rcon closed mid-packet')
            buf += chunk
        return buf[8:-2].decode('utf-8', 'replace')

    def cmd(self, text):
        self._send(self.CMD, text)
        return self._recv()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class LineLink:
    """JSON objects, one per line, both directions, with a re-read buffer.

    The gate streams a state per tick and may interleave world queries inside a
    tick, so reading cannot assume the next line answers the last write.
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = ''

    def send(self, obj):
        self.sock.sendall((json.dumps(obj) + '\n').encode('utf-8'))

    def readline(self, timeout=10.0):
        self.sock.settimeout(timeout)
        while '\n' not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.buf += chunk.decode('utf-8', 'replace')
        line, self.buf = self.buf.split('\n', 1)
        return line


class RemoteWorld:
    """`blocks` answered by the live server, column by column, cached per (x, z).

    The scan window is the whole column the game can hold (y_top down to the
    world's bottom), fetched in one query and cached, so the gate's 640-step
    column scan costs one round trip rather than 640.
    """

    def __init__(self, link, y_top=320, y_bottom=-64):
        self.link = link
        self.y_top, self.y_bottom = y_top, y_bottom
        self._cols = {}          # (x, z) -> {y: name} for non-air cells only
        self._id = 0
        self.queries = 0
        self.bytes_asked = 0
        self.blocks = Blocks(self)

    # -- fetching --------------------------------------------------------
    def _fetch(self, x, z):
        self._id += 1
        qid = self._id
        self.link.send({'q': {'id': qid, 'x': int(x), 'z': int(z),
                              'y0': self.y_top, 'y1': self.y_bottom}})
        self.queries += 1
        for _ in range(1000):
            line = self.link.readline()
            if line is None:
                raise ConnectionError('body closed while answering a column query')
            if not line or line[0] != 'C':
                continue                      # a stray state line, skip
            msg = json.loads(line[1:])
            if msg.get('id') != qid:
                continue
            col = {}
            for y, name in msg.get('cells') or ():
                col[int(y)] = name
            self._cols[(int(x), int(z))] = col
            self.bytes_asked += len(msg.get('cells') or ())
            return col
        raise ConnectionError('no answer to column query after 1000 lines')

    def name_at(self, cell):
        x, y, z = int(math.floor(cell[0])), int(math.floor(cell[1])), int(math.floor(cell[2]))
        col = self._cols.get((x, z))
        if col is None:
            col = self._fetch(x, z)
        return col.get(y)

    def invalidate(self, x=None, z=None):
        if x is None:
            self._cols.clear()
        else:
            self._cols.pop((int(math.floor(x)), int(math.floor(z))), None)

    # -- shape the gate expects ------------------------------------------
    def solid(self, cell):
        import voxel
        return voxel.is_solid(self.name_at(cell))


class Blocks:
    """The two methods embodied.py uses: .get(cell) and (rarely) membership."""

    def __init__(self, world):
        self.world = world

    def get(self, cell, default=None):
        got = self.world.name_at(cell)
        return default if got is None else got

    def __contains__(self, cell):
        return self.world.name_at(cell) is not None

    def __getitem__(self, cell):
        got = self.world.name_at(cell)
        if got is None:
            raise KeyError(cell)
        return got
