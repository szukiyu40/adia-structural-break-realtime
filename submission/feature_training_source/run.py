"""Run the fixed staged comparison; keep artifacts and clean own scratch."""
from integration_common import *
import subprocess
import shutil

def call(name, *args):
    subprocess.run([str(PYTHON), '-u', str(OUT / name), *map(str, args)], check=True)

def main():
    call('prepare.py'); call('verify.py')
    results = []
    for fold in range(5):
        if not (OUT / f'fold{fold}/result.json').exists():
            (OUT / f'fold{fold}').mkdir(exist_ok=True)
            for inner in (0, 1):
                if not predictor_path(fold, inner).exists():
                    code = ('import sys; from pathlib import Path; '
                            f'sys.path.insert(0,{str(OUT / "predictor_source")!r}); '
                            'import common; '
                            f'common.CACHE=Path({str(ARRAYS)!r}); '
                            'import train_predictor; '
                            f'train_predictor.train({fold},{inner},steps=800,out={str(predictor_path(fold,inner).parent)!r})')
                    subprocess.run([str(PYTHON), '-u', '-c', code], check=True)
                call('extract.py', '--fold', fold, '--inner', inner)
            call('fit.py', '--fold', fold)
        results.append(read(OUT / f'fold{fold}/result.json'))
        # This directory holds only reproducible features generated in this run.
        scratch = WORK / f'fold{fold}'
        if scratch.exists():
            shutil.rmtree(scratch)
        if sum(r['primary_delta'] <= 0 for r in results) >= 2:
            write(OUT / 'screen.json', {'status': 'stopped', 'reason': '4/5 positive folds impossible under fixed gate', 'completed_folds': len(results)})
            break
    call('report.py')

if __name__ == '__main__':
    main()
