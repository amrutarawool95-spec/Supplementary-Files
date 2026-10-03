import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
R=pd.read_csv('cv_results.csv')
fig,ax=plt.subplots(figsize=(6.2,3.1))
combos=[(f,m) for f in ['Length only','Antibody-style proxy','Full sequence panel'] for m in ['Ridge','RandomForest']]
for i,(f,m) in enumerate(combos):
    y=len(combos)-1-i
    g=R[(R.features==f)&(R.model==m)&(R.split.str.startswith('grouped'))].iloc[0]; r=R[(R.features==f)&(R.model==m)&(R.split.str.startswith('random'))].iloc[0]
    ax.errorbar(g.spearman,y,xerr=[[g.spearman-g.ci_lo],[g.ci_hi-g.spearman]],fmt='o',color='#1f5fa8',capsize=3,ms=6,label='Grouped by WT cluster (95% CI)' if i==0 else None)
    ax.plot(r.spearman,y,'D',color='#c0392b',ms=5,mfc='none',label='Random split (leaky)' if i==0 else None)
ax.set_yticks(range(len(combos))); ax.set_yticklabels([f'{f} / {m}' for f,m in combos][::-1]); ax.axvline(0,color='#999',lw=0.8)
ax.set_xlabel('Spearman correlation, predicted vs measured ΔG (out-of-fold)'); ax.set_xlim(-0.1,0.6); ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(0.35,-0.2),ncol=2)
plt.tight_layout(); plt.savefig('fig_cv.png',dpi=300); plt.close()
X=pd.read_csv('external_transfer.csv'); X=X[X.predictor.str.contains('predicted dG')].copy()
X['pred']=X.predictor.str.replace(' (predicted dG)','',regex=False)
combos=X[['scaffold','endpoint','n']].drop_duplicates().values.tolist()
preds=sorted(X.pred.unique()); cols=dict(zip(preds,['#1f5fa8','#7fb3e6','#d9822b','#f2c28b']))
fig,ax=plt.subplots(figsize=(6.4,4.3)); k=len(preds)
for i,(sc,ep,n) in enumerate(combos):
    for j,p in enumerate(preds):
        r=X[(X.scaffold==sc)&(X.endpoint==ep)&(X.pred==p)]
        if r.empty: continue
        r=r.iloc[0]; y=-(i+(j-(k-1)/2)*0.17)
        ax.errorbar(r.spearman,y,xerr=[[r.spearman-r.ci_lo],[r.ci_hi-r.spearman]],fmt='o',color=cols[p],ms=4,capsize=2,lw=1,label=p if i==0 else None)
ax.set_yticks([-i for i in range(len(combos))]); ax.set_yticklabels([f'{sc}: {ep} (n={n})' for sc,ep,n in combos],fontsize=7)
ax.axvline(0,color='#999',lw=0.8); ax.set_xlabel('Spearman correlation: predicted ΔG vs measured endpoint (95% CI)'); ax.legend(frameon=False,fontsize=6.5,loc='upper center',bbox_to_anchor=(0.35,-0.14),ncol=2)
plt.tight_layout(); plt.savefig('fig_ext.png',dpi=300); plt.close()
      
