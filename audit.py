"""Refresh quality status from already completed independent metrics; never edit metrics."""
from pathlib import Path
import json,argparse
def audit(folder):
    folder=Path(folder)
    metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    basic=metrics['dark_leak_captured']<.02 and metrics['rmse_relative']<.35 and metrics['tir']==0 and metrics['captured_flux']>.98
    fine=metrics['thin_flux_ratio'] is None or .8<=metrics['thin_flux_ratio']<=1.2
    status.update(basic_imaging_pass=basic,thin_detail_pass=fine,optical_quality_pass=basic and fine,
                  quality_revision='basic_plus_native_thin_flux_v2',cnc_release=False)
    (folder/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(folder,status)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='+');a=p.parse_args()
    for folder in a.folder:audit(folder)
