#!/usr/bin/env python3
"""Verify our renderer with Unity's incompatible Embree ABI already global.

Usage: python3 Tools/verify_embree_isolation.py /path/to/Unity/Editor/libembree.so
This runs in a separate process; it does not open or change a Unity project.
"""
import ctypes
from pathlib import Path
import runpy
import sys

unity_embree = Path(sys.argv[1]).resolve()
foreign_library = ctypes.CDLL(str(unity_embree), mode=ctypes.RTLD_GLOBAL)
assert foreign_library.rtcNewScene
print('Loaded Unity Embree globally; checking the isolated renderer', flush=True)
runpy.run_path(str(Path(__file__).with_name('verify_native.py')), run_name='__main__')
print('PASS Unity Embree and our Embree render in the same process without symbol collisions')
