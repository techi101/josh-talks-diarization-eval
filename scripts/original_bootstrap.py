"""Paired session bootstrap (the significance test from the design) on the small AMI run, strict settings."""
import pandas as pd, numpy as np
R=pd.concat([pd.read_csv('small_run_results.csv'),pd.read_csv('sortformer_results.csv'),pd.read_csv('pyannote_results.csv')]); R=R[R.setting.str.startswith('strict')]
tot={m:sum(float(l.split()[4]) for l in open(f'ami/{m}.rttm')) for m in R.meeting.unique()}  # reference speech time (s, overlap counted per speaker approx)
R['T']=R.meeting.map(tot); R['E']=R.DER/100*R['T']; R['rec']=R.meeting+'|'+R.mic
P=R.pivot_table(index='rec',columns='system',values=['E','T'])
sy=['S1 VAD + single speaker','S2 VAD + ECAPA + AHC (auto count)','S3 same, told true count','Sortformer v2, 30 s latency (offline-like)','Sortformer v2, 1.04 s latency (live)','pyannote community-1 (auto count)','pyannote community-1, told true count']
short={sy[0]:'S1',sy[1]:'S2',sy[2]:'S3',sy[3]:'Sortformer offline',sy[4]:'Sortformer live',sy[5]:'pyannote',sy[6]:'pyannote told count'}
rng=np.random.default_rng(0); n=len(P); B=10000; out=[]
for a,b in [(sy[1],sy[0]),(sy[2],sy[1]),(sy[2],sy[0]),(sy[3],sy[2]),(sy[4],sy[3]),(sy[4],sy[2]),(sy[5],sy[3]),(sy[6],sy[5])]:
    d=[]
    for _ in range(B):
        i=rng.integers(0,n,n); s=P.iloc[i]
        d.append(100*(s['E'][a].sum()/s['T'][a].sum()-s['E'][b].sum()/s['T'][b].sum()))
    d=np.array(d); obs=100*(P['E'][a].sum()/P['T'][a].sum()-P['E'][b].sum()/P['T'][b].sum())
    out.append(dict(comparison=f'{short[a]} minus {short[b]}',observed=round(obs,1),ci_lo=round(np.percentile(d,2.5),1),ci_hi=round(np.percentile(d,97.5),1),p_better=round((d<0).mean(),3)))
o=pd.DataFrame(out); print(o.to_string(index=False)); o.to_csv('bootstrap_small.csv',index=False)
