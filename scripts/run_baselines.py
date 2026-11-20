"""Small diarization eval, v2 (crash-safe).
3 AMI test meetings x 2 mics (close-talk headset mix vs far-field array ch1); 1 AMI dev meeting (ES2011a) used ONLY to pick
the clustering threshold. Systems: S1 Silero VAD + one speaker; S2 Silero VAD + ECAPA + agglomerative clustering
(threshold tuned on dev); S3 same embeddings, told the true speaker count. Scored with pyannote.metrics in 4 settings."""
import subprocess, os, sys, json, platform
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'speechbrain', 'silero-vad', 'pyannote.metrics', 'soundfile'], check=True)
TEST = ['ES2004a', 'IS1009a', 'TS3003a']; DEVM = ['ES2011a']; MICS = ['Mix-Headset', 'Array1-01']
os.makedirs('ami', exist_ok=True)
for m, split in [(x, 'test') for x in TEST] + [(x, 'dev') for x in DEVM]:
    subprocess.run(['wget', '-q', '-O', f'ami/{m}.rttm', f'https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/{split}/{m}.rttm'], check=True)
    for c in MICS:
        subprocess.run(['wget', '-q', '-O', f'ami/{m}.{c}.wav', f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{c}.wav'], check=True)
import numpy as np, soundfile as sf, torch, pandas as pd
from silero_vad import load_silero_vad, get_speech_timestamps
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
torch.set_num_threads(4)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', DEV, flush=True)
vad = load_silero_vad(); enc = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir='sb_ecapa', run_opts={'device': DEV})
WIN, HOP = 1.5, 0.75
def load_rttm(p):
    a = Annotation()
    for l in open(p):
        q = l.split(); a[Segment(float(q[3]), float(q[3]) + float(q[4]))] = q[7]
    return a
def read16(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    assert sr == 16000; return x
def embed(x):
    ts = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=16000, return_seconds=True)
    speech = [(t['start'], t['end']) for t in ts]; wins = []
    for s, e in speech:
        a = s
        while a < e:
            b = min(a + WIN, e)
            if b - a >= 0.3: wins.append((a, b))
            a += HOP
    embs = []
    for i in range(0, len(wins), 64):
        batch = [x[int(a * 16000):int(b * 16000)] for a, b in wins[i:i + 64]]; L = max(len(c) for c in batch)
        w = torch.zeros(len(batch), L); lens = torch.tensor([len(c) / L for c in batch])
        for j, c in enumerate(batch): w[j, :len(c)] = torch.from_numpy(c)
        with torch.no_grad(): embs.append(enc.encode_batch(w.to(DEV), lens.to(DEV)).squeeze(1).cpu().numpy())
    E = np.concatenate(embs); E /= np.linalg.norm(E, axis=1, keepdims=True)
    return speech, np.array(wins), E
def one_speaker(speech):
    a = Annotation()
    for s, e in speech: a[Segment(s, e)] = 'A'
    return a
def cluster(speech, wins, E, thr=None, n=None):
    lab = AgglomerativeClustering(n_clusters=n, distance_threshold=None if n else thr, metric='cosine', linkage='average').fit(E).labels_
    # each speech region is cut at window midpoints; each piece takes its window's label (full VAD coverage kept)
    out = Annotation(); k = 0
    for s, e in speech:
        idx = [i for i in range(len(wins)) if wins[i][0] >= s - 1e-6 and wins[i][1] <= e + 1e-6]
        if not idx: continue
        bounds = [s] + [(wins[i][0] + wins[i + 1][1]) / 2 for i in idx[:-1]] + [e]
        for j, i in enumerate(idx):
            if bounds[j + 1] > bounds[j]: out[Segment(bounds[j], bounds[j + 1])] = f'spk{lab[i]}'
    return out.support(collar=0.0)
# 1) embeddings for every file, saved as we go
cache = {}
for m in DEVM + TEST:
    for mic in MICS:
        sp, wins, E = embed(read16(f'ami/{m}.{mic}.wav'))
        cache[(m, mic)] = (sp, wins, E)
        np.savez(f'emb_{m}_{mic}.npz', speech=np.array(sp), wins=wins, E=E); print('embedded', m, mic, len(wins), flush=True)
# 2) pick the threshold on the dev meeting only
grid = np.round(np.arange(0.30, 0.96, 0.05), 2); dev_rows = []
for thr in grid:
    ders = []
    for mic in MICS:
        ref = load_rttm(f'ami/{DEVM[0]}.rttm'); sp, wins, E = cache[(DEVM[0], mic)]
        h = cluster(sp, wins, E, thr=thr); ders.append(DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, h))
    dev_rows.append(dict(thr=thr, dev_DER=100 * np.mean(ders)))
dev = pd.DataFrame(dev_rows); dev.to_csv('threshold_dev.csv', index=False); THR = float(dev.loc[dev.dev_DER.idxmin(), 'thr'])
print(dev.round(2).to_string(index=False)); print('chosen threshold', THR, flush=True)
# 3) score the test meetings
SETTINGS = {'strict (0 collar, overlap scored)': dict(collar=0.0, skip_overlap=False), '0.25 s collar, overlap scored': dict(collar=0.5, skip_overlap=False),
            '0 collar, overlap skipped': dict(collar=0.0, skip_overlap=True), '0.25 s collar, overlap skipped (old NIST style)': dict(collar=0.5, skip_overlap=True)}
rows = []
for m in TEST:
    ref = load_rttm(f'ami/{m}.rttm'); n = len(ref.labels())
    for mic in MICS:
        sp, wins, E = cache[(m, mic)]
        for name, h in [('S1 VAD + single speaker', one_speaker(sp)), ('S2 VAD + ECAPA + AHC (auto count)', cluster(sp, wins, E, thr=THR)), ('S3 same, told true count', cluster(sp, wins, E, n=n))]:
            jer = 100 * JaccardErrorRate(collar=0.0)(ref, h)
            for sname, kw in SETTINGS.items():
                d = DiarizationErrorRate(**kw)(ref, h, detailed=True); T = d['total']
                rows.append(dict(meeting=m, mic=mic, system=name, setting=sname, DER=100 * d['diarization error rate'], miss=100 * d['missed detection'] / T,
                                 fa=100 * d['false alarm'] / T, conf=100 * d['confusion'] / T, ref_spk=n, hyp_spk=len(h.labels()), JER=jer))
        pd.DataFrame(rows).to_csv('small_run_results.csv', index=False); print(m, mic, 'scored', flush=True)
df = pd.DataFrame(rows)
env = dict(python=platform.python_version(), torch=torch.__version__, speechbrain=__import__('speechbrain').__version__,
           silero_vad=__import__('importlib.metadata').metadata.version('silero-vad'), pyannote_metrics=__import__('importlib.metadata').metadata.version('pyannote.metrics'),
           ecapa='speechbrain/spkrec-ecapa-voxceleb', window_s=WIN, hop_s=HOP, ahc='average linkage, cosine', threshold=THR, threshold_tuned_on='AMI dev ES2011a (both mics)',
           vad='silero default params', reference='pyannote AMI-diarization-setup only_words RTTM', device=DEV)
json.dump(env, open('small_run_env.json', 'w'), indent=1)
print(df[df.setting.str.startswith('strict')].round(1).to_string(index=False))
print(df.groupby(['system', 'setting']).DER.mean().unstack().round(1).to_string())
import shutil; shutil.rmtree('ami', ignore_errors=True); shutil.rmtree('sb_ecapa', ignore_errors=True)
