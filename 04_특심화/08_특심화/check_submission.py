"""실습08: 새 프로젝트를 만들지 않고 07의 동일 프로젝트를 검증합니다."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'course_support.py').is_file())
# 배포 형태에 따라 프로젝트가 예전 이름(ex07_compare) 또는 번호 폴더(07_특심화)에 있을 수 있습니다.
# 실제로 존재하는 경로를 선택하면 폴더 이름이 달라도 같은 검증기를 재사용할 수 있습니다.
OLD_PROJECT=ROOT/'ex07_compare/03_심화/day6_compare'
CURRENT_PROJECT=ROOT/'07_특심화/day6_compare'
PROJECT=OLD_PROJECT if OLD_PROJECT.is_dir() else CURRENT_PROJECT
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true')
    p.add_argument('--trust-local',action='store_true')
    a=p.parse_args()
    args=[sys.executable,str(PROJECT/'verify_outputs.py')]
    if a.restore: args.append('--restore')
    if a.trust_local: args.append('--trust-local')
    return subprocess.run(args,check=False).returncode
if __name__=='__main__': raise SystemExit(main())
