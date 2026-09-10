"""CPU-only synthetic process tree; never imports Torch or a DLSS runtime."""
import subprocess
import sys
import time

mode = sys.argv[1]
if mode in ('orphan', 'tree_hang'):
    child = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time;time.sleep(120)'],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    print(f'CHILD={child.pid}', flush=True)
if mode in ('hang', 'tree_hang'):
    time.sleep(120)
elif mode == 'flood':
    for _ in range(100):
        sys.stdout.buffer.write(b'o'*8192)
        sys.stderr.buffer.write(b'e'*8192)
elif mode == 'error':
    raise RuntimeError('deliberate fixture failure')
elif mode == 'complete':
    print('OWNED_TASK_COMPLETE', flush=True)
