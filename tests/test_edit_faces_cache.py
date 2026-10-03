# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The faces pass caches the surroundings of an open group on the placements
epoch, like the edges' head: every frame of a Move drag inside a group bumps
the version, and the sync walked every placement's chunk for faces that had
not changed (~25 ms a frame beside 576 small groups). The buffers it hands
to the GPU must be byte-identical to the uncached walk, and anything that
changes the surroundings must still show."""
from __future__ import annotations

import re

import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D as V
from PySide6.QtWidgets import QApplication

from core.group import Group
from core.mesh import Mesh

_app = QApplication.instance() or QApplication([])


def _box(x, y, z, s=1.0, attrs=None):
    m = Mesh()
    c = [V(x + dx * s, y + dy * s, z + dz * s)
         for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    for q in ([0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4],
              [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]):
        f = m.add_face([c[i] for i in q])
        if attrs and f is not None:
            f.attrs.update(attrs)
    return m


def _scene(win):
    sc = win.viewport.scene
    blocks = []
    for b in range(3):
        floors = []
        for f in range(3):
            boxes = []
            for i in range(4):
                attrs = ({"color": (0.8, 0.2, 0.2)} if i == 1 else
                         {"color": (0.2, 0.4, 0.9), "opacity": 0.5}
                         if i == 2 else None)
                g = Group(_box(b * 6 + i * 1.4, 0, f * 1.4, 1.0, attrs),
                          name=f"box {b}.{f}.{i}")
                g.component = False
                boxes.append(g)
            fl = Group(Mesh(), name=f"floor {b}.{f}")
            fl.component = False
            fl.adopt(boxes)
            for g in boxes:
                g.owner = fl
            floors.append(fl)
        bl = Group(Mesh(), name=f"block {b}")
        bl.component = False
        bl.adopt(floors)
        for fl in floors:
            fl.owner = bl
        blocks.append(bl)
    lone = Group(_box(-4, 0, 0), name="lone")
    lone.component = False
    sc.groups.extend(blocks + [lone])
    sc.version += 1
    return sc, blocks, lone


def _sync(vp):
    """Run the sync with a fake upload; every buffer's bytes, by slot, plus
    the spans and splits the draw passes read."""
    sent = {}

    def fake_upload(_vbo, slot, parts, empty=24):
        data = b"".join(parts)
        sent[slot] = data
        return len(data)
    vp._upload_vbo = fake_upload
    import views.viewport as vpmod
    src = open(vpmod.__file__, encoding="utf-8").read()
    for name in set(re.findall(r"self\.(_\w+_vbo)\b", src)):
        if not hasattr(vp, name):
            setattr(vp, name, None)
    vp._tick = getattr(vp, "_tick", 0) + 1
    vp._edges_version = -1               # sync even at the same version
    vp._sync_edges()
    return (sent, vp._face_spans, vp._edit_split_f, vp._edit_split_db,
            vp._tcol_runs, vp._dback_spans, vp._tex_runs)


@pytest.fixture
def win():
    from views.main_window import MainWindow
    w = MainWindow()
    try:
        yield w
    finally:
        w._saved_version = w.viewport.scene.version
        w.close()


def _enter_deep(vp, blocks):
    block = blocks[1]
    floor = block.children[1]
    vp.begin_group_edit(block)
    vp.begin_group_edit(next(c for c in block.children
                             if c.name == floor.name))
    box = next(c for c in vp.scene.edit_group.children
               if c.name.endswith(".1"))
    vp.begin_group_edit(box)
    return vp.scene.mesh


def _drag(sc, mesh, k):
    v = next(iter(mesh.vertices))
    mesh.move_vertex(v, V(0.05 * k, 0.0, 0.02 * k))
    sc.version += 1                      # what Move's live preview does


@pytest.mark.parametrize("depth", ["top", "inside"])
def test_cached_surroundings_send_the_same_buffers(win, monkeypatch, depth):
    import views.viewport as vv
    vp = win.viewport
    sc, blocks, lone = _scene(win)
    mesh = _enter_deep(vp, blocks) if depth == "inside" else None
    if mesh is None:
        sc.mesh.add_face([V(-8, 0, 0), V(-7, 0, 0), V(-7, 1, 0)])
        sc.version += 1
        mesh = sc.mesh
    for k in range(1, 4):
        _drag(sc, mesh, k)
        monkeypatch.setattr(vv, "_NO_EDIT_FACE_CACHE", False)
        cached = _sync(vp)
        monkeypatch.setattr(vv, "_NO_EDIT_FACE_CACHE", True)
        walked = _sync(vp)
        assert cached == walked


def test_a_drag_inside_does_not_walk_the_surroundings(win, monkeypatch):
    import views.viewport as vv
    monkeypatch.setattr(vv, "_NO_EDIT_FACE_CACHE", False)
    vp = win.viewport
    sc, blocks, lone = _scene(win)
    mesh = _enter_deep(vp, blocks)
    _drag(sc, mesh, 1)
    _sync(vp)                                       # fills the cache
    asked = []
    chunk = vp._group_chunk

    def counting(g):
        name = getattr(g, "name", None)      # prototype wrappers have none
        if name is not None:
            asked.append(name)
        return chunk(g)
    monkeypatch.setattr(vp, "_group_chunk", counting)
    _drag(sc, mesh, 2)
    _sync(vp)
    # Only the open box (and nothing of the 35 placements around it).
    assert len(set(asked)) <= 1


def test_a_change_in_the_surroundings_still_shows(win, monkeypatch):
    import views.viewport as vv
    monkeypatch.setattr(vv, "_NO_EDIT_FACE_CACHE", False)
    vp = win.viewport
    sc, blocks, lone = _scene(win)
    _enter_deep(vp, blocks)
    before = _sync(vp)[0]["faces"]
    lone.hidden = True                   # Hide on a surrounding object
    sc.version += 1
    after = _sync(vp)
    assert len(after[0]["faces"]) < len(before)
    monkeypatch.setattr(vv, "_NO_EDIT_FACE_CACHE", True)
    assert _sync(vp) == after
