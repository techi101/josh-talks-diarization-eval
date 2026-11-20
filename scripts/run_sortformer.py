"""NVIDIA Streaming Sortformer v2 (nvidia/diar_streaming_sortformer_4spk-v2, open weights) on the same 3 AMI test meetings x 2 mics,
at two latency settings from the model card. Scored exactly like the baselines (pyannote.metrics, 4 settings). Saves after every file."""
import subprocess, sys, os, json, time
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'Cython', 'packaging'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nemo_toolkit[asr]', 'pyannote.metrics', 'soundfile'], check=True)
TEST = ['ES2004a', 'IS1009a', 'TS3003a']; MICS = ['Mix-Headset', 'Array1-01']
os.makedirs('ami', exist_ok=True)
for m in TEST:
    subprocess.run(['wget', '-q', '-O', f'ami/{m}.rttm', f'https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/{m}.rttm'], check=True)
    for c in MICS:
        subprocess.run(['wget', '-q', '-O', f'ami/{m}.{c}.wav', f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{c}.wav'], check=True)
import torch, pandas as pd, soundfile as sf, numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
import importlib.metadata as md
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True)
model = SortformerEncLabelModel.from_pretrained('nvidia/diar_streaming_sortformer_4spk-v2').to(dev).eval()
CFG = {'Sortformer v2, 30 s latency (offline-like)': dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300, spkcache_len=188),
       'Sortformer v2, 1.04 s latency (live)': dict(chunk_len=6, chunk_right_context=7, fifo_len=188, spkcache_update_period=144, spkcache_len=188)}
SETTINGS = {'strict (0 collar, overlap scored)': dict(collar=0.0, skip_overlap=False), '0.25 s collar, overlap scored': dict(collar=0.5, skip_overlap=False),
            '0 collar, overlap skipped': dict(collar=0.0, skip_overlap=True), '0.25 s collar, overlap skipped (old NIST style)': dict(collar=0.5, skip_overlap=True)}
def load_rttm(p):
    a = Annotation()
    for l in open(p):
        q = l.split(); a[Segment(float(q[3]), float(q[3]) + float(q[4]))] = q[7]
    return a
def to_ann(segs):
    a = Annotation()
    for s in segs:
        if isinstance(s, str): p = s.replace(',', ' ').split(); st, en, lab = float(p[0]), float(p[1]), p[2]
        else: st, en, lab = float(s[0]), float(s[1]), str(s[2])
        if en > st: a[Segment(st, en)] = lab
    return a
rows = []; timing = []
for cname, cfg in CFG.items():
    for k, v in cfg.items(): setattr(model.sortformer_modules, k, v)
    for m in TEST:
        ref = load_rttm(f'ami/{m}.rttm'); n = len(ref.labels())
        for mic in MICS:
            path = f'ami/{m}.{mic}.wav'; dur = sf.info(path).duration; t0 = time.time()
            with torch.no_grad(): segs = model.diarize(audio=[path], batch_size=1)[0]
            rtf = (time.time() - t0) / dur; h = to_ann(segs)
            jer = 100 * JaccardErrorRate(collar=0.0)(ref, h)
            for sname, kw in SETTINGS.items():
                d = DiarizationErrorRate(**kw)(ref, h, detailed=True); T = d['total']
                rows.append(dict(meeting=m, mic=mic, system=cname, setting=sname, DER=100 * d['diarization error rate'], miss=100 * d['missed detection'] / T,
                                 fa=100 * d['false alarm'] / T, conf=100 * d['confusion'] / T, ref_spk=n, hyp_spk=len(h.labels()), JER=jer, rtf=rtf))
            pd.DataFrame(rows).to_csv('sortformer_results.csv', index=False); print(cname, m, mic, 'DER strict', round(rows[-4]['DER'], 1), 'rtf', round(rtf, 3), flush=True)
json.dump(dict(model='nvidia/diar_streaming_sortformer_4spk-v2', nemo=md.version('nemo_toolkit'), torch=torch.__version__, pyannote_metrics=md.version('pyannote.metrics'),
               device=dev, configs=CFG, python=sys.version.split()[0]), open('sortformer_env.json', 'w'), indent=1)
df = pd.DataFrame(rows); print(df.groupby(['system', 'setting']).DER.mean().unstack().round(1).to_string())
import shutil; shutil.rmtree('ami', ignore_errors=True)
