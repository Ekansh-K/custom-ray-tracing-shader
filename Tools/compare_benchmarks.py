#!/usr/bin/env python3
"""Compare matching standalone runs and require unchanged per-backend images."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('before',type=Path)
parser.add_argument('after',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
before=json.loads((args.before/'results.json').read_text())
after=json.loads((args.after/'results.json').read_text())
assert before['snapshot_sha256']==after['snapshot_sha256'],'Scene snapshots differ'
assert before['measured_passes']==after['measured_passes'],'Measured pass counts differ'
assert before['warmup_passes']==after['warmup_passes'],'Warm-up pass counts differ'
assert before['omp_threads']==after['omp_threads'],'CPU worker configuration differs'

def key(case): return (case['width'],case['height'],case['bounces'],case['rays_per_pixel'])
old={key(case):case for case in before['results']}
assert set(old)=={key(case) for case in after['results']},'Benchmark cases differ'
rows=[]
for case in after['results']:
    previous=old[key(case)]
    assert len(previous['backends'])==len(case['backends'])
    for a,b in zip(previous['backends'],case['backends']):
        assert a['backend']==b['backend'] and a['samples_per_pixel']==b['samples_per_pixel'] and a['cpu_threads']==b['cpu_threads'],'Workload differs'
        assert a['raw_image_sha256']==b['raw_image_sha256'],'Image changed: '+b['backend']+' '+str(key(case))
        rows.append({'width':case['width'],'height':case['height'],'bounces':case['bounces'],
            'rays_per_pixel':case['rays_per_pixel'],'samples_per_pixel':b['samples_per_pixel'],
            'backend':b['backend'],'before_compute_ms':a['compute_median_ms'],'after_compute_ms':b['compute_median_ms'],
            'compute_speedup':a['compute_median_ms']/b['compute_median_ms'],
            'before_call_ms':a['call_median_ms'],'after_call_ms':b['call_median_ms'],
            'call_speedup':a['call_median_ms']/b['call_median_ms'],'image_identical':True})
        print(f'{case["width"]}x{case["height"]}, {case["bounces"]} hits, {b["backend"]}: '
              f'{a["compute_median_ms"]:.3f} -> {b["compute_median_ms"]:.3f} ms '
              f'({a["compute_median_ms"]/b["compute_median_ms"]:.2f}x), image identical')
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps({'snapshot_sha256':before['snapshot_sha256'],
    'triangles':before['triangles'],'measured_passes':before['measured_passes'],
    'method':'Same scene snapshot, sample counts, bounces and CPU worker count. Per-backend output hashes must match exactly. Native timings exclude Unity texture upload and display.',
    'results':rows},indent=2)+'\n')
