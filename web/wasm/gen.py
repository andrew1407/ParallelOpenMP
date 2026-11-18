"""Make adjustable copies of the lab sources that need it (into web/wasm/gen/).

Each replacement must match exactly once, so a change in a lab source fails the build
instead of silently compiling something else.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / 'gen'

EDITS = {
    '2-lab/Parallel_OpenMP.cpp': ('lab2.cpp', [
        ('const int threads = 8;', 'const int threads = lab2_threads;'),
        ('const int aLen = 16000;', 'const int aLen = lab2_len;'),
        # 16000-element stack arrays become heap buffers so the size can grow
        ('int a[aLen];', 'std::vector<int> aBuf(aLen); int* a = aBuf.data();'),
        ('int b[aLen];', 'std::vector<int> bBuf(aLen); int* b = bBuf.data();'),
    ]),
    '3-lab/Parallel_OpenMP.cpp': ('lab3.cpp', [
        ('#define SIZE 500', 'int SIZE = 500;'),
    ]),
}

OUT.mkdir(exist_ok=True)
for src, (dst, edits) in EDITS.items():
    text = (ROOT / src).read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f'{src}: expected exactly one "{old}"')
        text = text.replace(old, new)
    (OUT / dst).write_text(f'// Generated from {src} by web/wasm/gen.py. Do not edit.\n#line 1 "{src}"\n' + text)
    print(f'gen/{dst} <- {src}')
