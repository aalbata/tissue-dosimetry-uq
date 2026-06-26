import numpy as np, pandas as pd, json, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from cole_cole import load_table, cole_cole_params, eps_sigma
from uq import pennes_peak, thermal
warnings.filterwarnings('ignore'); np.random.seed(3)
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':0.3,'savefig.bbox':'tight'})
R=json.load(open('revision_results.json'))
data,cols=load_table(); names=data['Tissue'].astype(str)
G=lambda i,c: pd.to_numeric(data.iloc[i,c],errors='coerce')
TIS=['Skin','Fat','Muscle']; TH=[0.0015,0.005,None]
base=[names[names.str.fullmatch(nm,case=False)].index[0] for nm in TIS]
FREQS=[0.9e9,2.45e9,5.8e9]; FLAB=['0.9','2.45','5.8']; S=50.0

def sigq(f,seed=0):
    feats=[97,2,7,12,17]; X=[];y=[]
    for i in range(len(data)):
        er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
        if np.isnan(sg) or sg<=0: continue
        X.append([G(i,c) for c in feats]); y.append(sg)
    X=pd.DataFrame(X).apply(pd.to_numeric,errors='coerce'); X=X.fillna(X.median()); y=np.array(y)
    rng=np.random.RandomState(seed); idx=rng.permutation(len(y)); ntr=int(0.7*len(y))
    m=ExtraTreesRegressor(n_estimators=400,random_state=seed).fit(X.iloc[idx[:ntr]],y[idx[:ntr]])
    return float(np.quantile(np.abs(y[idx[ntr:]]-m.predict(X.iloc[idx[ntr:]])),0.90))

def draw_specs(f,rng,q,sig_nom):
    sp=[]
    for li in range(3):
        i=base[li]; er,_=eps_sigma(f,cole_cole_params(data.iloc[i]))
        def rt(key):
            lo=thermal(i,key,'min'); hi=thermal(i,key,'max'); a=thermal(i,key,'avg')
            return rng.uniform(min(lo,hi),max(lo,hi)) if hi>lo else a
        sg=max(sig_nom[li]+rng.uniform(-q,q),1e-3)
        sp.append((TH[li],dict(rho=rt('rho'),c=0.0,k=rt('k'),perf=rt('perf'),eps_r=er,sigma=sg)))
    return sp

# ---- depth-profile bands at 2.45 GHz ----
f=2.45e9; q=sigq(f); sig_nom=[eps_sigma(f,cole_cole_params(data.iloc[i]))[1] for i in base]
spn=[(TH[li],dict(rho=float(G(base[li],2)),c=0.0,k=thermal(base[li],'k','avg'),perf=thermal(base[li],'perf','avg'),
      eps_r=eps_sigma(f,cole_cole_params(data.iloc[base[li]]))[0],sigma=sig_nom[li])) for li in range(3)]
_,_,x,SARn,dTn,_=pennes_peak(f,spn,S); xmm=x*1000
rng=np.random.RandomState(5); M=300; SARs=[];dTs=[]
for _ in range(M):
    _,_,_,si,di,_=pennes_peak(f,draw_specs(f,rng,q,sig_nom),S); SARs.append(si); dTs.append(di)
SARs=np.array(SARs); dTs=np.array(dTs)
SARlo,SARhi=np.percentile(SARs,5,0),np.percentile(SARs,95,0)
dTlo,dThi=np.percentile(dTs,5,0),np.percentile(dTs,95,0)

fig,ax=plt.subplots(figsize=(5.2,3.4)); m=xmm<=20
ax.fill_between(xmm[m],SARlo[m],SARhi[m],color='#3b7dd8',alpha=0.25,label='central 90% interval')
ax.plot(xmm[m],SARn[m],color='#16324f',lw=1.8,label='nominal (tabulated)')
for e in [1.5,6.5]: ax.axvline(e,color='0.5',ls='--',lw=0.8)
yl=ax.get_ylim()[1]
ax.text(0.4,yl*0.9,'skin',fontsize=8); ax.text(3,yl*0.9,'fat',fontsize=8); ax.text(11,yl*0.9,'muscle',fontsize=8)
ax.set_xlabel('Depth (mm)'); ax.set_ylabel('SAR (W/kg)'); ax.set_title('Local SAR vs depth, 2.45 GHz, 50 W/m$^2$')
ax.legend(frameon=False,fontsize=8); fig.savefig('fig_sar_depth.pdf')

fig,ax=plt.subplots(figsize=(5.2,3.4)); m=xmm<=30
ax.fill_between(xmm[m],dTlo[m],dThi[m],color='#d8662b',alpha=0.25,label='central 90% interval')
ax.plot(xmm[m],dTn[m],color='#7a2f12',lw=1.8,label='nominal (tabulated)')
for e in [1.5,6.5]: ax.axvline(e,color='0.5',ls='--',lw=0.8)
ax.set_xlabel('Depth (mm)'); ax.set_ylabel('Temperature rise $\\Delta T$ (\u00B0C)'); ax.set_title('Steady-state $\\Delta T$ vs depth, 2.45 GHz, 50 W/m$^2$')
ax.legend(frameon=False,fontsize=8); fig.savefig('fig_dT_depth.pdf')

# ---- spread ratio (from json bands) ----
sar_fac=[R['nominal'][fl]['SAR_band'][1]/R['nominal'][fl]['SAR_band'][0] for fl in FLAB]
dT_fac=[R['nominal'][fl]['dT_band'][1]/R['nominal'][fl]['dT_band'][0] for fl in FLAB]
fig,ax=plt.subplots(figsize=(5.0,3.2)); xp=np.arange(3); w=0.35
ax.bar(xp-w/2,sar_fac,w,color='#3b7dd8',label='peak SAR'); ax.bar(xp+w/2,dT_fac,w,color='#d8662b',label='peak $\\Delta T$')
ax.set_xticks(xp); ax.set_xticklabels(FLAB); ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('90% interval width (upper/lower)'); ax.axhline(1,color='0.4',lw=0.8)
ax.set_title('Dosimetric uncertainty from tissue-property ranges'); ax.legend(frameon=False,fontsize=8)
fig.savefig('fig_spread.pdf')

# ---- Sobol property (ST with CI), SAR + dT ----
PROP=['Conductivity','Perfusion','ThermalCond','Density']
col={'Conductivity':'#3b7dd8','Perfusion':'#d8662b','ThermalCond':'#5aa469','Density':'#9b8bd0'}
lab={'Conductivity':'Conductivity $\\sigma$','Perfusion':'Perfusion','ThermalCond':'Thermal cond. $k$','Density':'Density $\\rho$'}
def get(fl,out,key):
    d=R['sobol'][fl]['property'][out]; idx={g:i for i,g in enumerate(d['groups'])}
    return [d[key][idx[p]] for p in PROP]
fig,axes=plt.subplots(1,2,figsize=(8.4,3.4),sharey=True); xp=np.arange(3); w=0.2
for ax,out,ttl in zip(axes,['SAR','dT'],['Peak SAR','Peak $\\Delta T$']):
    for j,p in enumerate(PROP):
        st=np.clip(get(FLAB[0] if False else None,out,'ST'),0,1.1) if False else None
        vals=[np.clip(R['sobol'][fl]['property'][out]['ST'][R['sobol'][fl]['property'][out]['groups'].index(p)],0,1.15) for fl in FLAB]
        errs=[R['sobol'][fl]['property'][out]['ST_conf'][R['sobol'][fl]['property'][out]['groups'].index(p)] for fl in FLAB]
        ax.bar(xp+(j-1.5)*w, vals, w, yerr=errs, capsize=2, error_kw={'lw':0.7}, color=col[p], label=lab[p])
    ax.set_xticks(xp); ax.set_xticklabels(FLAB); ax.set_xlabel('Frequency (GHz)'); ax.set_title(ttl); ax.set_ylim(0,1.2)
axes[0].set_ylabel('Sobol total-effect index $S_T$')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Which property drives dosimetric uncertainty (Sobol $S_T$, 95% CI)',y=1.02,fontsize=11)
fig.savefig('fig_sensitivity_property.pdf')

# ---- Sobol tissue (dT) ----
TC={'Skin':'#e0a458','Fat':'#d8662b','Muscle':'#7a2f12'}
fig,ax=plt.subplots(figsize=(5.0,3.3)); w=0.25
for j,nm in enumerate(TIS):
    vals=[np.clip(R['sobol'][fl]['layer']['dT']['ST'][R['sobol'][fl]['layer']['dT']['groups'].index(nm)],0,1.15) for fl in FLAB]
    errs=[R['sobol'][fl]['layer']['dT']['ST_conf'][R['sobol'][fl]['layer']['dT']['groups'].index(nm)] for fl in FLAB]
    ax.bar(xp+(j-1)*w, vals, w, yerr=errs, capsize=2, error_kw={'lw':0.7}, color=TC[nm], label=nm)
ax.set_xticks(xp); ax.set_xticklabels(FLAB); ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Sobol total-effect index $S_T$'); ax.set_title('Which tissue drives the $\\Delta T$ uncertainty')
ax.legend(frameon=False,fontsize=9); fig.savefig('fig_sensitivity_tissue.pdf')

# ---- conformal validation: predicted vs tabulated at 2.45 GHz ----
feats=[97,2,7,12,17]; X=[];y=[]
for i in range(len(data)):
    er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
    if np.isnan(sg) or sg<=0: continue
    X.append([G(i,c) for c in feats]); y.append(sg)
X=pd.DataFrame(X).apply(pd.to_numeric,errors='coerce'); X=X.fillna(X.median()).values; y=np.array(y); n=len(y)
rng=np.random.RandomState(0); idx=rng.permutation(n); ntr=int(0.5*n); nca=int(0.25*n)
tr,ca,te=idx[:ntr],idx[ntr:ntr+nca],idx[ntr+nca:]
m=ExtraTreesRegressor(n_estimators=300,random_state=0).fit(X[tr],y[tr])
qv=np.quantile(np.abs(y[ca]-m.predict(X[ca])),0.90); pred=m.predict(X[te])
fig,ax=plt.subplots(figsize=(4.7,3.6))
lim=[0, max(y.max(),pred.max())*1.05]
ax.fill_between(lim,[l-qv for l in lim],[l+qv for l in lim],color='#3b7dd8',alpha=0.15,label='±q (90%% band), q=%.2f S/m'%qv)
ax.plot(lim,lim,'k--',lw=0.8)
ax.errorbar(y[te],pred,yerr=qv,fmt='o',ms=4,color='#16324f',ecolor='#9bb7d8',elinewidth=0.8,capsize=0,label='held-out tissues')
ax.set_xlim(lim); ax.set_ylim(lim)
cov=R['conformal']['2.45']['coverage_mean']
ax.set_xlabel('Tabulated conductivity (S/m)'); ax.set_ylabel('Predicted conductivity (S/m)')
ax.set_title('Conformal interval check, 2.45 GHz\nempirical coverage %.2f (target 0.90)'%cov)
ax.legend(frameon=False,fontsize=7.5,loc='upper left'); fig.savefig('fig_conformal.pdf')
print('figures regenerated')
