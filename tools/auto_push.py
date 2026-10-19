"""
Auto-sync this repository to GitHub.

Watches the working tree; once changes have been quiet for QUIET_SECONDS (so a half-finished
edit is not pushed), it commits them with a message listing the changed files and pushes to
origin/main. Commits made by hand that are not pushed yet are pushed too.

Start:   pythonw tools\\auto_push.py          (runs in the background, no window)
Status:  see tools\\auto_push.log
Stop:    tools\\auto_push.py --stop
Pause:   create a file named .autopush-pause in the repo root (delete it to resume)

Installed to start at Windows logon by tools\\install_auto_push.bat.
"""
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, 'tools', 'auto_push.log')
LOCK = os.path.join(REPO, 'tools', 'auto_push.pid')
PAUSE = os.path.join(REPO, '.autopush-pause')
BRANCH = 'main'
POLL_SECONDS = 15
QUIET_SECONDS = 120              # changes must be untouched this long before they are committed
MAX_FILE_MB = 50                 # GitHub rejects files over 100 MB; stay well below
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0  # CREATE_NO_WINDOW: no console flashes under pythonw


def log(message: str) -> None:
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {message}'
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > 1_000_000:  # keep the log small
            os.replace(LOG, LOG + '.old')
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def git(*args, check=False) -> subprocess.CompletedProcess:
    result = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', creationflags=NO_WINDOW)
    if check and result.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed: {(result.stderr or result.stdout).strip()}')
    return result


def git_busy() -> bool:
    """True while another git command, merge or rebase is in progress."""
    git_dir = os.path.join(REPO, '.git')
    return any(os.path.exists(os.path.join(git_dir, name))
               for name in ('index.lock', 'MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD'))


def snapshot() -> str:
    """Fingerprint of the current uncommitted changes (paths + modification times)."""
    status = git('status', '--porcelain', '--untracked-files=all').stdout
    parts = []
    for line in status.splitlines():
        path = line[3:].split(' -> ')[-1].strip('"')
        full = os.path.join(REPO, path)
        mtime = os.path.getmtime(full) if os.path.exists(full) else 0
        parts.append(f'{line}|{mtime}')
    return '\n'.join(parts)


def commit_message(paths) -> str:
    names = [p.split('/')[-1] for p in paths]
    shown = ', '.join(names[:3]) + (f' and {len(names) - 3} more' if len(names) > 3 else '')
    return f'Update {shown}\n\nAuto-synced changes:\n' + '\n'.join(f'- {p}' for p in paths)


def commit_changes() -> bool:
    git('add', '-A', check=True)
    staged = [p for p in git('diff', '--cached', '--name-only').stdout.splitlines() if p]
    for path in list(staged):
        full = os.path.join(REPO, path)
        if os.path.isfile(full) and os.path.getsize(full) > MAX_FILE_MB * 1024 * 1024:
            git('reset', '-q', 'HEAD', '--', path)
            staged.remove(path)
            log(f'SKIPPED {path}: larger than {MAX_FILE_MB} MB (add it to .gitignore)')
    if not staged:
        return False
    git('commit', '-q', '-m', commit_message(staged), check=True)
    log(f'Committed {len(staged)} file(s): {", ".join(staged[:5])}{" ..." if len(staged) > 5 else ""}')
    return True


def push_if_ahead() -> None:
    if git('fetch', '-q', 'origin', BRANCH).returncode != 0:
        log('Fetch failed (offline?), will retry')
        return
    ahead = git('rev-list', '--count', f'origin/{BRANCH}..HEAD').stdout.strip()
    behind = git('rev-list', '--count', f'HEAD..origin/{BRANCH}').stdout.strip()
    if behind not in ('', '0'):
        rebase = git('rebase', '-q', f'origin/{BRANCH}')
        if rebase.returncode != 0:
            git('rebase', '--abort')
            log('Rebase onto origin failed (conflict). Resolve it by hand: git pull --rebase')
            return
    if ahead in ('', '0') and behind in ('', '0'):
        return
    pushed = git('push', '-q', 'origin', f'HEAD:{BRANCH}')
    if pushed.returncode == 0:
        log(f'Pushed to origin/{BRANCH} ({git("rev-parse", "--short", "HEAD").stdout.strip()})')
    else:
        log(f'Push failed, will retry: {(pushed.stderr or pushed.stdout).strip()[:300]}')


def already_running() -> bool:
    try:
        pid = int(open(LOCK).read().strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    if os.name == 'nt':
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True,
                             creationflags=NO_WINDOW).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop() -> None:
    try:
        pid = int(open(LOCK).read().strip())
    except (OSError, ValueError):
        print('auto_push is not running')
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True, creationflags=NO_WINDOW)
    else:
        os.kill(pid, 15)
    os.remove(LOCK)
    log('Stopped')
    print('auto_push stopped')


def main() -> None:
    if '--stop' in sys.argv:
        stop()
        return
    if already_running():
        return
    with open(LOCK, 'w') as f:
        f.write(str(os.getpid()))
    log(f'Started, watching {REPO}')
    last_snapshot, quiet_since = None, time.time()
    while True:
        try:
            if os.path.exists(PAUSE) or git_busy():
                time.sleep(POLL_SECONDS)
                continue
            current = snapshot()
            if current != last_snapshot:
                last_snapshot, quiet_since = current, time.time()
            elif current and time.time() - quiet_since >= QUIET_SECONDS:
                if commit_changes():
                    push_if_ahead()
                last_snapshot = snapshot()
            elif not current and git('rev-list', '--count', f'origin/{BRANCH}..HEAD').stdout.strip() not in ('', '0'):
                push_if_ahead()  # hand-made commits that are not pushed yet (checked offline first)
        except Exception as error:  # keep running; the next cycle retries
            log(f'Error: {error}')
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
