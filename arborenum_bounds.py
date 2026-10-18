import ctypes
import os
import re
import sys
import tempfile
from contextlib import contextmanager

_libc = ctypes.CDLL(None)


_BOUND_PATTERNS = [
    re.compile(r'^Objective bound(?: \(user-set\))?: (\d+)'),
    re.compile(r'^First objective bound: (\d+)'),
    re.compile(r'^Extending from objective bound \d+ to (\d+)'),
    re.compile(r'with objective bound (\d+)$'),
    re.compile(r'converged at budget (\d+)'),
    re.compile(r'keeping completed objective bound (\d+)'),
]


@contextmanager
def capture_native_stdout():
    sys.stdout.flush()
    captured = {'text': ''}
    saved = os.dup(1)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield captured
        finally:
            _libc.fflush(None)
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            captured['text'] = tmp.read().decode(errors='replace')
            sys.stdout.write(captured['text'])
            sys.stdout.flush()


def parse_bounds(text):
    out = {'reference_objective': None, 'objective_bound': None, 'min_objective': None}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'Best objective: (\d+)', line)
        if m:
            out['reference_objective'] = int(m.group(1))
        m = re.match(r'Minimum objective: (\d+)', line)
        if m:
            out['min_objective'] = int(m.group(1))
        for pattern in _BOUND_PATTERNS:
            m = pattern.search(line)
            if m:
                out['objective_bound'] = int(m.group(1))
    return out
