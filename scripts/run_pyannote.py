"""pyannote speaker-diarization-community-1 (open weights, gated: needs HF_TOKEN Kaggle secret) on the same 3 AMI test meetings x 2 mics
as the baselines and Sortformer. Default run (count not given) + oracle-count run. Scored identically (pyannote.metrics, 4 settings)."""
import subprocess, sys, os, json, time
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as e:
    HF = None
if not HF:
    print('NO HF_TOKEN SECRET: open Edit > Add-ons > Secrets, add HF_TOKEN, attach it, then Save & Run All'); sys.exit(1)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyannote.audio>=4.0', 'pyannote.metrics', 'soundfile'], check=True)
TEST = ['ES2004a', 'IS1009a', 'TS3003a']; MICS = ['Mix-Headset', 'Array1-01']
os.makedirs('ami', exist_ok=True)
for m in TEST:
    subprocess.run(['wget', '-q', '-O', f'ami/{m}.rttm', f'https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/{m}.rttm'], check=True)
    for c in MICS:
        subprocess.run(['wget', '-q', '-O', f'ami/{m}.{c}.wav', f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{c}.wav'], check=True)
import torch, pandas as pd, soundfile as sf, numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
import importlib.metadata as md
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True)
pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=HF).to(torch.device(dev))
SETTINGS = {'strict (0 collar, overlap scored)': dict(collar=0.0, skip_overlap=False), '0.25 s collar, overlap scored': dict(collar=0.5, skip_overlap=False),
            '0 collar, overlap skipped': dict(collar=0.0, skip_overlap=True), '0.25 s collar, overlap skipped (old NIST style)': dict(collar=0.5, skip_overlap=True)}
RUNS = {'pyannote community-1 (auto count)': {}, 'pyannote community-1, told true count': {'num_speakers': 4}}
def load_rttm(p):
    a = Annotation()
    for l in open(p):
        q = l.split(); a[Segment(float(q[3]), float(q[3]) + float(q[4]))] = q[7]
    return a
rows = []
for m in TEST:
    ref = load_rttm(f'ami/{m}.rttm'); n = len(ref.labels())
    for mic in MICS:
        x, sr = sf.read(f'ami/{m}.{mic}.wav', dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        audio = {'waveform': torch.from_numpy(x)[None], 'sample_rate': sr}; dur = len(x) / sr
        for rname, kw in RUNS.items():
            t0 = time.time(); out = pipe(audio, **kw); rtf = (time.time() - t0) / dur
            h = out.speaker_diarization if hasattr(out, 'speaker_diarization') else out
            jer = 100 * JaccardErrorRate(collar=0.0)(ref, h)
            for sname, s in SETTINGS.items():
                d = DiarizationErrorRate(**s)(ref, h, detailed=True); T = d['total']
                rows.append(dict(meeting=m, mic=mic, system=rname, setting=sname, DER=100 * d['diarization error rate'], miss=100 * d['missed detection'] / T,
                                 fa=100 * d['false alarm'] / T, conf=100 * d['confusion'] / T, ref_spk=n, hyp_spk=len(h.labels()), JER=jer, rtf=rtf))
            pd.DataFrame(rows).to_csv('pyannote_results.csv', index=False); print(rname, m, mic, 'DER strict', round(rows[-4]['DER'], 1), 'rtf', round(rtf, 3), flush=True)
json.dump(dict(model='pyannote/speaker-diarization-community-1', pyannote_audio=md.version('pyannote.audio'), torch=torch.__version__,
               pyannote_metrics=md.version('pyannote.metrics'), device=dev, python=sys.version.split()[0]), open('pyannote_env.json', 'w'), indent=1)
df = pd.DataFrame(rows); print(df.groupby(['system', 'setting']).DER.mean().unstack().round(1).to_string())
import shutil; shutil.rmtree('ami', ignore_errors=True)
