"""Check actual release archives without importing System1 or optional libraries."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def main(directory):
    wheels = list(directory.glob('*.whl'))
    sources = list(directory.glob('*.tar.gz'))
    assert len(wheels) == len(sources) == 1, 'Expected one wheel and one source archive'
    wheel, source = wheels[0], sources[0]
    assert wheel.stat().st_size < 2 * 1024**2, 'Wheel unexpectedly exceeds 2 MiB'
    assert source.stat().st_size < 25 * 1024**2, 'Source archive unexpectedly exceeds 25 MiB'
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert {'system1/compiler.py', 'system1/teaching.py', 'system1/proto/system1.proto',
            'reflex/__init__.py', 'reflex/teaching.py', 'reflex/proto/system1.proto'} <= wheel_names
    assert not any(name.startswith(('benchmarks/', 'examples/')) for name in wheel_names)
    with tarfile.open(source, 'r:gz') as archive:
        source_names = {member.name.split('/', 1)[1] for member in archive.getmembers() if '/' in member.name}
    assert {'src/system1/compiler.py', 'examples/teaching_by_doing/app.py',
            'examples/teaching_by_doing/index.html', 'examples/teaching_by_doing/documents.json',
            'src/system1/teaching.py', 'src/reflex/teaching.py',
            'docs/guides/correcting_skills.md',
            'benchmarks/quality/instinct_gauntlet_report.md',
            'benchmarks/quality/drop_in_and_adaptation_report.md',
            'benchmarks/quality/poke_env_showdown_report.md',
            'benchmarks/quality/document_workflow/run.py',
            'benchmarks/quality/document_workflow/PROTOCOL.md',
            'benchmarks/quality/document_workflow/results/summary.json',
            'benchmarks/quality/document_workflow/results/evidence.zip',
            'benchmarks/quality/datasets/public_splits/official-test.json',
            'benchmarks/quality/datasets/public_splits/boundary-teaching-requests.json'} <= source_names
    print(f'Checked wheel ({wheel.stat().st_size:,} bytes) and source archive ({source.stat().st_size:,} bytes).')


if __name__ == '__main__':
    import shutil
    import subprocess
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=None)
    args = parser.parse_args()
    if args.directory is not None:
        main(args.directory)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            if shutil.which('uv'):
                subprocess.run(['uv', 'build', '--out-dir', str(tmp_path)], check=True)
            else:
                subprocess.run([sys.executable, '-m', 'build', '--outdir', str(tmp_path)], check=True)
            shutil.rmtree(Path('build'), ignore_errors=True)
            main(tmp_path)


