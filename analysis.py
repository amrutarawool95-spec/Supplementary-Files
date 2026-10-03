import numpy as np, pandas as pd, re, json, warnings
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
U='./'
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
AA='ACDEFGHIKLMNPQRSTVWY'
def feats(s):
    pa=ProteinAnalysis(s); L=len(s)
    kd=np.array([KD[a] for a in s]); w=7
    win=np.convolve(kd,np.ones(w)/w,mode='valid') if L>=w else kd
    ang=np.deg2rad(100*np.arange(L)); hm=np.hypot((kd*np.cos(ang)).sum(),(kd*np.sin(ang)).sum())/L
    q=pa.charge_at_pH(7.4); h,t,e=pa.secondary_structure_fraction()
    d={'length':L,'GRAVY':kd.mean(),'net_charge':q,'abs_charge_per_res':abs(q)/L,'max_hydrophobic_window':win.max(),
       'hydrophobic_moment':hm,'pI':pa.isoelectric_point(),'aromaticity':pa.aromaticity(),'helix_frac':h,'turn_frac':t,'sheet_frac':e}
    for a in AA: d['f_'+a]=s.count(a)/L
    return d
SETS={'Length only':['length'],
      'Antibody-style proxy':['GRAVY','net_charge','abs_charge_per_res','max_hydrophobic_window'],
      'Full sequence panel':None}
def mk(name):
    return make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,20))) if name=='Ridge' else RandomForestRegressor(300,min_samples_leaf=3,random_state=0,n_jobs=-1)
# ---- data
t=pd.read_csv(U+'tsuboyama_wt_40_72.csv').drop_duplicates('exact_sequence_group').reset_index(drop=True)
t['natural']=t.domain_id.str.match(r'^[0-9][A-Za-z0-9]{3}\.pdb')
t['topology']=t.domain_id.str.extract(r'^(EEHEE|EHEE|HHH|HEEH)_')[0]
F=pd.DataFrame([feats(s) for s in t.sequence_aa]); y=t.deltaG_source_mean_kcal_mol.values; g=t.source_WT_cluster.values
SETS['Full sequence panel']=list(F.columns)
print('Tsuboyama n',len(t),'natural',t.natural.sum(),'designed',(~t.natural).sum(),'clusters',len(set(g)),'topologies',t.topology.value_counts().to_dict())
def cluster_boot_sp(pred,y,g,B=1000,seed=1):
    rng=np.random.default_rng(seed); ug=np.unique(g); idx={k:np.where(g==k)[0] for k in ug}; out=[]
    for _ in range(B):
        s=np.concatenate([idx[k] for k in rng.choice(ug,len(ug))]); 
        if np.ptp(pred[s])>0: out.append(spearmanr(pred[s],y[s])[0])
    return np.percentile(out,[2.5,97.5])
def oof(X,y,g,model,grouped=True,reps=5,k=10):
    P=np.zeros((reps,len(y))); ug=np.unique(g)
    for r in range(reps):
        rng=np.random.default_rng(100+r)
        if grouped:
            fold={u:i%k for i,u in enumerate(rng.permutation(ug))}; f=np.array([fold[x] for x in g])
        else:
            f=rng.permutation(len(y))%k
        for i in range(k):
            tr,te=f!=i,f==i; m=mk(model).fit(X[tr],y[tr]); P[r,te]=m.predict(X[te])
    return P.mean(0)
rows=[]; preds={}
for sn,cols in SETS.items():
    X=F[cols].values
    for mn in ['Ridge','RandomForest']:
        for gr in [True,False]:
            p=oof(X,y,g,mn,gr); preds[(sn,mn,gr)]=p
            sp=spearmanr(p,y)[0]; ci=cluster_boot_sp(p,y,g) if gr else (np.nan,np.nan)
            nat=t.natural.values
            rows.append(dict(features=sn,model=mn,split='grouped by WT cluster' if gr else 'random (leaky)',spearman=sp,ci_lo=ci[0],ci_hi=ci[1],
              spearman_natural=spearmanr(p[nat],y[nat])[0],spearman_designed=spearmanr(p[~nat],y[~nat])[0],rmse=np.sqrt(np.mean((p-y)**2))))
R=pd.DataFrame(rows); R.to_csv('cv_results.csv',index=False); print(R.round(3).to_string())
# leave-topology-out (designed)
rows=[]
for top in ['EEHEE','EHEE','HHH','HEEH']:
    te=(t.topology==top).values; tr=~te
    for sn,cols in SETS.items():
        for mn in ['Ridge','RandomForest']:
            X=F[cols].values; m=mk(mn).fit(X[tr],y[tr]); p=m.predict(X[te])
            rows.append(dict(held_out_topology=top,n=te.sum(),features=sn,model=mn,spearman=spearmanr(p,y[te])[0]))
LT=pd.DataFrame(rows); LT.to_csv('leave_topology_out.csv',index=False); print(LT.round(3).to_string())
# ---- external: Nielsen
n=pd.read_csv(U+'nielsen_2024_external.csv').drop_duplicates('exact_sequence_group').reset_index(drop=True)
Fn=pd.DataFrame([feats(s) for s in n.sequence_aa])
ENDP={'Average TL':'Protease-assay score (TL)','Average sGFP':'Split-GFP score','CD Tm (degC)':'CD Tm','Dot Blot Avg (ug/mL)':'Dot-blot yield','SDS PAGE (mg/mL)':'SDS-PAGE yield'}
def boot(a,b,B=2000,seed=2):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(B):
        i=rng.integers(0,len(a),len(a)); 
        if np.ptp(a[i])>0 and np.ptp(b[i])>0: out.append(spearmanr(a[i],b[i])[0])
    return np.percentile(out,[2.5,97.5])
rows=[]
models={}
for sn,cols in SETS.items():
    for mn in ['Ridge','RandomForest']: models[(sn,mn)]=(cols,mk(mn).fit(F[cols].values,y))
for sc in ['affibody','fibronectin']:
    sub=n.scaffold==sc
    for ec,en in ENDP.items():
        ok=sub&n[ec].notna()
        if ok.sum()<10: continue
        v=n.loc[ok,ec].values
        for (sn,mn),(cols,m) in models.items():
            if sn=='Length only': continue
            p=m.predict(Fn.loc[ok,cols].values)
            if np.ptp(p)==0: continue
            r=spearmanr(p,v)[0]; ci=boot(p,v); rows.append(dict(scaffold=sc,endpoint=en,n=int(ok.sum()),predictor=f'{sn} / {mn} (predicted dG)',spearman=r,ci_lo=ci[0],ci_hi=ci[1]))
        for d in ['GRAVY','net_charge']:
            a=Fn.loc[ok,d].values; r=spearmanr(a,v)[0]; ci=boot(a,v); rows.append(dict(scaffold=sc,endpoint=en,n=int(ok.sum()),predictor=d+' (raw descriptor)',spearman=r,ci_lo=ci[0],ci_hi=ci[1]))
X=pd.DataFrame(rows); X.to_csv('external_transfer.csv',index=False); print(X.round(3).to_string())
# counts for text
print('Nielsen n',len(n),n.scaffold.value_counts().to_dict())
t[['domain_id','natural','topology']].assign(**F[['GRAVY','net_charge']]).to_csv('tsuboyama_descriptors_subset.csv',index=False)
pd.concat([t[['domain_id','sequence_aa','source_WT_cluster','natural','topology','deltaG_source_mean_kcal_mol']],F],axis=1).to_csv('tsuboyama_descriptors.csv',index=False)
pd.concat([n[['scaffold','source_record_id','sequence_aa','Average TL','Average sGFP','CD Tm (degC)','Dot Blot Avg (ug/mL)','SDS PAGE (mg/mL)']],Fn],axis=1).to_csv('nielsen_descriptors.csv',index=False)
