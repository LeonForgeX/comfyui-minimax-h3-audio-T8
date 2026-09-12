"""Isolated tests: no ComfyUI/model imports or publication."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

GUARD = Path(__file__).with_name('check_no_roadmap.py')
spec = importlib.util.spec_from_file_location('guard', GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardTests(unittest.TestCase):
    def test_case_and_nested_paths(self):
        self.assertEqual(guard.forbidden(['ROADMAP.md', 'a/b/RoadMap.MD', 'docs/RELEASE.md']),
                         ['ROADMAP.md', 'a/b/RoadMap.MD'])

    def test_deleted_file_remains_blocked_in_history(self):
        import sys
        with tempfile.TemporaryDirectory(prefix='h3-roadmap-guard-') as folder:
            def git(*args):
                return subprocess.run(['git', *args], cwd=folder, check=True, capture_output=True)
            git('init')
            git('config', 'user.email', 'fixture@example.invalid')
            git('config', 'user.name', 'Fixture')
            Path(folder, 'README.md').write_text('fixture')
            git('add', 'README.md')
            git('commit', '-m', 'clean')
            def run(*args, data=None):
                return subprocess.run([sys.executable, str(GUARD), *args], cwd=folder,
                                      input=data, capture_output=True, text=True)
            self.assertEqual(run('--history', 'HEAD').returncode, 0)
            Path(folder, 'roadmap.md').write_text('synthetic fixture only')
            git('add', 'roadmap.md')
            self.assertEqual(run().returncode, 1)
            git('commit', '-m', 'synthetic forbidden file')
            git('rm', 'roadmap.md')
            git('commit', '-m', 'remove from tree only')
            self.assertEqual(run().returncode, 0)
            self.assertEqual(run('--history', 'HEAD').returncode, 1)
            sha = git('rev-parse', 'HEAD').stdout.decode().strip()
            data = f'refs/heads/test {sha} refs/heads/test ' + '0' * 40 + '\n'
            self.assertEqual(run('--pre-push', data=data).returncode, 1)
            self.assertEqual(run('--pre-push', data='bad protocol\n').returncode, 1)
            self.assertEqual(run('--pre-push', data='x ' + '0' * 40 + ' y ' + sha + '\n').returncode, 0)


if __name__ == '__main__':
    unittest.main()
