import numpy as np, pandas as pd, json, warnings
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from sklearn.ensemble import ExtraTreesRegressor
from cole_cole import load_table, cole_cole_params, eps_sigma, EPS0
from dosimetry import slab_SAR
from uq import pennes_peak, thermal
warnings.filterwarnings('ignore'); np.random.seed(12345)
MU0=4e-7*np.pi

data,cols=load_table(); names=data['Tissue'].astype(str)
G=lambda i,c: pd.to_numeric(data.iloc[i,c],errors='coerce')
TIS=['Skin','Fat','Muscle']; TH=[0.0015,0.005,None]
base=[names[names.str.fullmatch(nm,case=False)].index[0] for nm in TIS]
FREQS=[0.9e9,2.45e9,5.8e9]; FLAB=['0.9','2.45','5.8']
S_INC=50.0
out={'energy':{}, 'sobol':{}, 'conformal':{}, 'nominal':{}}

# ---------- 1. Energy audit at all three frequencies (adaptive deep tail) ----------
def nominal_layers(f):
    L=[]
    for li,nm in enumerate(TIS):
        i=base[li]; er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
        L.append(dict(d=TH[li],eps_r=er,sigma=sg,rho=float(G(i,2))))
    return L
print("ENERGY AUDIT (adaptive tail)")
for f,fl in zip(FREQS,FLAB):
    L=nominal_layers(f)
    # penetration depth in muscle -> tail length = 12 skin depths
    w=2*np.pi*f; epc=EPS0*L[-1]['eps_r']-1j*L[-1]['sigma']/w
    alpha=np.real(1j*w*np.sqrt(MU0*epc)); tail=min(0.6,max(0.06,12.0/alpha))
    xs,sar,Gam=slab_SAR(f,L,S_inc=S_INC,tail_len=tail)
    # rho profile
    edges=np.cumsum([0,L[0]['d'],L[1]['d'],tail]); rho=np.zeros_like(xs)
    for li in range(3):
        m=(xs>=edges[li])&(xs<=edges[li+1]); rho[m]=L[li]['rho']
    Pabs=np.trapezoid(rho*sar,xs); Pin=(1-abs(Gam)**2)*S_INC
    epsE=abs(Pabs-Pin)/Pin
    out['energy'][fl]={'eps_E':float(epsE),'Gamma':float(abs(Gam)),'tail_m':float(tail)}
    print("  %s GHz: eps_E=%.4f  |Gamma|=%.3f  tail=%.2f m"%(fl,epsE,abs(Gam),tail))

# ---------- nominal + 90% interval (Monte Carlo) for reference ----------
def sigma_q(f, seed=0):
    feats=[97,2,7,12,17]; X=[]; y=[]
    for i in range(len(data)):
        er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
        if np.isnan(sg) or sg<=0: continue
        X.append([G(i,c) for c in feats]); y.append(sg)
    X=pd.DataFrame(X).apply(pd.to_numeric,errors='coerce'); X=X.fillna(X.median()); y=np.array(y)
    rng=np.random.RandomState(seed); idx=rng.permutation(len(y)); ntr=int(0.7*len(y))
    m=ExtraTreesRegressor(n_estimators=400,random_state=seed).fit(X.iloc[idx[:ntr]],y[idx[:ntr]])
    res=np.abs(y[idx[ntr:]]-m.predict(X.iloc[idx[ntr:]])); return float(np.quantile(res,0.90))

# ---------- model wrapper for Sobol ----------
def make_eval(f):
    er=[eps_sigma(f,cole_cole_params(data.iloc[i]))[0] for i in base]
    def model(v):
        sp=[]
        for li in range(3):
            sp.append((TH[li],dict(rho=v[9+li],c=0.0,k=v[6+li],perf=v[3+li],eps_r=er[li],sigma=max(v[li],1e-3))))
        a,b,*_=pennes_peak(f,sp,S_INC); return a,b
    return model

def bounds_for(f):
    q=sigma_q(f); b=[]; names_v=[]; groups_p=[]; groups_l=[]
    sig_nom=[eps_sigma(f,cole_cole_params(data.iloc[i]))[1] for i in base]
    # sigma (3)
    for li in range(3):
        b.append([max(sig_nom[li]-q,1e-3), sig_nom[li]+q]); names_v.append('sig_%s'%TIS[li]); groups_p.append('Conductivity'); groups_l.append(TIS[li])
    # perf, k, rho (3 each)
    for key,gname in [('perf','Perfusion'),('k','ThermalCond'),('rho','Density')]:
        for li in range(3):
            i=base[li]; lo=thermal(i,key,'min'); hi=thermal(i,key,'max'); a=thermal(i,key,'avg')
            if not (hi>lo): lo,hi=a*0.999,a*1.001
            b.append([lo,hi]); names_v.append('%s_%s'%(key,TIS[li])); groups_p.append(gname); groups_l.append(TIS[li])
    return b,names_v,groups_p,groups_l,q

def run_sobol(f, grouping, N=256):
    b,nv,gp,gl,q=bounds_for(f)
    groups = gp if grouping=='property' else gl
    problem={'num_vars':12,'names':nv,'bounds':b,'groups':groups}
    X=sobol_sample.sample(problem,N,calc_second_order=False)
    model=make_eval(f)
    Y=np.array([model(x) for x in X])
    res={}
    for j,out_name in enumerate(['SAR','dT']):
        Si=sobol_analyze.analyze(problem,Y[:,j],calc_second_order=False,num_resamples=200)
        gnames=Si['names'] if 'names' in Si else sorted(set(groups),key=groups.index)
        res[out_name]={'groups':list(gnames),
                       'S1':[float(x) for x in Si['S1']],'S1_conf':[float(x) for x in Si['S1_conf']],
                       'ST':[float(x) for x in Si['ST']],'ST_conf':[float(x) for x in Si['ST_conf']]}
    res['n_eval']=len(X); res['q']=q
    return res

print("\nSOBOL INDICES (N=256, total-effect ST shown)")
for f,fl in zip(FREQS,FLAB):
    out['sobol'][fl]={}
    for grouping in ['property','layer']:
        r=run_sobol(f,grouping); out['sobol'][fl][grouping]=r
    p=out['sobol'][fl]['property']
    print("  %s GHz  SAR ST:"%fl, {g:round(v,2) for g,v in zip(p['SAR']['groups'],p['SAR']['ST'])})
    print("           dT  ST:", {g:round(v,2) for g,v in zip(p['dT']['groups'],p['dT']['ST'])})

# ---------- nominal + 90% MC band ----------
print("\nNOMINAL + 90% INTERVAL")
for f,fl in zip(FREQS,FLAB):
    Ln=[(TH[li],dict(rho=float(G(base[li],2)),c=0.0,k=thermal(base[li],'k','avg'),
         perf=thermal(base[li],'perf','avg'),
         eps_r=eps_sigma(f,cole_cole_params(data.iloc[base[li]]))[0],
         sigma=eps_sigma(f,cole_cole_params(data.iloc[base[li]]))[1])) for li in range(3)]
    sar0,dT0,*_=pennes_peak(f,Ln,S_INC)
    q=sigma_q(f); model=make_eval(f); rng=np.random.RandomState(7); M=2000; sars=np.empty(M); dts=np.empty(M)
    sig_nom=[eps_sigma(f,cole_cole_params(data.iloc[i]))[1] for i in base]
    for m in range(M):
        v=np.empty(12)
        for li in range(3): v[li]=max(sig_nom[li]+rng.uniform(-q,q),1e-3)
        for off,key in [(3,'perf'),(6,'k'),(9,'rho')]:
            for li in range(3):
                i=base[li]; lo=thermal(i,key,'min'); hi=thermal(i,key,'max'); a=thermal(i,key,'avg')
                v[off+li]=rng.uniform(min(lo,hi),max(lo,hi)) if hi>lo else a
        sars[m],dts[m]=model(v)
    band=lambda z:[float(np.percentile(z,5)),float(np.percentile(z,95))]
    out['nominal'][fl]={'SAR_nom':float(sar0),'SAR_band':band(sars),'dT_nom':float(dT0),'dT_band':band(dts),
                        'SAR_p50':float(np.percentile(sars,50)),'dT_p50':float(np.percentile(dts,50))}
    print("  %s GHz  SAR nom=%.2f band=%s | dT nom=%.2f band=%s"%(fl,sar0,[round(x,2) for x in band(sars)],dT0,[round(x,3) for x in band(dts)]))

# ---------- conformal validation (repeated train/calib/test) ----------
print("\nCONFORMAL VALIDATION (50 splits, target coverage 0.90)")
for f,fl in zip(FREQS,FLAB):
    feats=[97,2,7,12,17]; X=[]; y=[]
    for i in range(len(data)):
        er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
        if np.isnan(sg) or sg<=0: continue
        X.append([G(i,c) for c in feats]); y.append(sg)
    X=pd.DataFrame(X).apply(pd.to_numeric,errors='coerce'); X=X.fillna(X.median()).values; y=np.array(y); n=len(y)
    covs=[]; qs=[]
    for s in range(50):
        rng=np.random.RandomState(s); idx=rng.permutation(n)
        ntr=int(0.5*n); nca=int(0.25*n)
        tr,ca,te=idx[:ntr],idx[ntr:ntr+nca],idx[ntr+nca:]
        m=ExtraTreesRegressor(n_estimators=300,random_state=s).fit(X[tr],y[tr])
        q=np.quantile(np.abs(y[ca]-m.predict(X[ca])),0.90)
        pred=m.predict(X[te]); cov=np.mean(np.abs(y[te]-pred)<=q)
        covs.append(cov); qs.append(q)
    out['conformal'][fl]={'coverage_mean':float(np.mean(covs)),'coverage_std':float(np.std(covs)),
                          'q_mean':float(np.mean(qs)),'q_std':float(np.std(qs)),'n':int(n)}
    print("  %s GHz  coverage=%.3f±%.3f  q=%.3f±%.3f S/m  (n=%d)"%(fl,np.mean(covs),np.std(covs),np.mean(qs),np.std(qs),n))

json.dump(out, open('revision_results.json','w'), indent=2)
print("\nsaved revision_results.json")
