import numpy as np, pandas as pd, warnings
from scipy.linalg import solve_banded
from sklearn.ensemble import ExtraTreesRegressor
from cole_cole import load_table, cole_cole_params, eps_sigma
from dosimetry import slab_SAR
warnings.filterwarnings('ignore'); np.random.seed(0)
RHO_B=1050.0; C_B=3617.0; T_ART=37.0; T_CORE=37.0; T_AIR=24.0; H_CONV=8.0

data,cols=load_table(); names=data['Tissue'].astype(str)
G=lambda i,c: pd.to_numeric(data.iloc[i,c],errors='coerce')
CM={'rho':(2,5,6),'c':(7,10,11),'k':(12,15,16),'perf':(17,20,21)}

def thermal(i, key, mode):  # mode avg/min/max with avg fallback
    a=G(i,CM[key][0]); v=G(i,CM[key][{'avg':0,'min':1,'max':2}[mode]])
    return a if np.isnan(v) else v

def pennes_peak(freq, specs, S_inc):
    ds=[t for t,_ in specs[:-1]]+[0.05]; Lt=sum(ds); N=600
    x=np.linspace(0,Lt,N); dx=x[1]-x[0]; edges=np.cumsum([0]+ds)
    lay=np.clip(np.searchsorted(edges, x, side='right')-1,0,len(ds)-1)
    P=[s[1] for s in specs]
    k=np.array([P[lay[i]]['k'] for i in range(N)]); rho=np.array([P[lay[i]]['rho'] for i in range(N)])
    perf=np.array([P[lay[i]]['perf'] for i in range(N)])
    Wb=RHO_B*C_B*perf*rho*1e-6/60.0
    lem=[dict(d=(ds[li] if li<len(ds)-1 else None),eps_r=P[li]['eps_r'],sigma=P[li]['sigma'],rho=P[li]['rho']) for li in range(len(ds))]
    xs,sar,Gam=slab_SAR(freq,lem,S_inc=S_inc); SAR=np.interp(x,xs,sar,left=sar[0],right=sar[-1])
    def solve(inc):
        ab=np.zeros((3,N)); b=np.zeros(N)
        ab[1,0]=k[0]/dx+H_CONV; ab[0,1]=-k[0]/dx; b[0]=H_CONV*T_AIR
        for i in range(1,N-1):
            ab[2,i-1]=k[i]/dx**2; ab[0,i+1]=k[i]/dx**2; ab[1,i]=-2*k[i]/dx**2-Wb[i]
            b[i]=-Wb[i]*T_ART-(rho[i]*SAR[i] if inc else 0.0)
        ab[1,N-1]=1.0; b[N-1]=T_CORE
        return solve_banded((1,1),ab,b)
    dT=solve(True)-solve(False)
    return sar.max(), dT.max(), x, SAR, (solve(True)-solve(False)), abs(Gam)

# ---- conformal completion model for sigma at frequency ----
def sigma_completion(freq):
    feats=[97,2,7,12,17]  # water content, density, heatcap, thermcond, perfusion
    sig=[]; X=[]
    for i in range(len(data)):
        er,sg=eps_sigma(freq, cole_cole_params(data.iloc[i]))
        if np.isnan(sg) or sg<=0: continue
        row=[G(i,c) for c in feats]
        X.append(row); sig.append(sg)
    X=pd.DataFrame(X).apply(pd.to_numeric,errors='coerce'); X=X.fillna(X.median()); y=np.array(sig)
    n=len(y); idx=np.random.permutation(n); ntr=int(0.7*n)
    tr,ca=idx[:ntr],idx[ntr:]
    m=ExtraTreesRegressor(n_estimators=400,random_state=0).fit(X.iloc[tr],y[tr])
    resid=np.abs(y[ca]-m.predict(X.iloc[ca])); q=np.quantile(resid,0.90)
    return q, len(y)

freqs=[0.9e9,2.45e9,5.8e9]
print('Conformal 90% sigma-interval half-width q (S/m) and n_train:')
qmap={}
for f in freqs:
    q,n=sigma_completion(f); qmap[f]=q
    print('  %.2f GHz: q=%.3f S/m (n=%d)'%(f/1e9,q,n))

def slab_specs(freq, mode='avg', sig_scale=(1,1,1)):
    out=[]; tis=['Skin','Fat','Muscle']; th=[0.0015,0.005,None]
    for t,nm,ss in zip(th,tis,sig_scale):
        i=names[names.str.fullmatch(nm,case=False)].index[0]
        er,sg=eps_sigma(freq, cole_cole_params(data.iloc[i]))
        d=dict(rho=thermal(i,'rho',mode),c=thermal(i,'c',mode),k=thermal(i,'k',mode),
               perf=thermal(i,'perf',mode),eps_r=er,sigma=max(sg*ss,1e-3))
        out.append((t,d))
    return out

print('\n=== Nominal vs worst-case envelope (skin/fat/muscle, S_inc=50 W/m^2) ===')
for f in freqs:
    specs=slab_specs(f,'avg')
    sar0,dT0,*_=pennes_peak(f,specs,50.0)
    # Monte Carlo: sigma per layer ~ Uniform(sigma*(1 +/- q/sigma)) using conformal q; thermal ~ U[min,max]
    M=400; sars=[]; dTs=[]
    base=[names[names.str.fullmatch(nm,case=False)].index[0] for nm in ['Skin','Fat','Muscle']]
    sig_nom=[eps_sigma(f,cole_cole_params(data.iloc[i]))[1] for i in base]
    for _ in range(M):
        ss=[1+np.random.uniform(-qmap[f],qmap[f])/s for s in sig_nom]
        sp=[]; 
        for li,(t,nm) in enumerate(zip([0.0015,0.005,None],['Skin','Fat','Muscle'])):
            i=base[li]; er,sg=eps_sigma(f,cole_cole_params(data.iloc[i]))
            def rt(key):
                a=thermal(i,key,'avg'); lo=thermal(i,key,'min'); hi=thermal(i,key,'max')
                return np.random.uniform(min(lo,hi),max(lo,hi)) if (lo!=hi) else a
            sp.append((t,dict(rho=rt('rho'),c=rt('c'),k=rt('k'),perf=rt('perf'),eps_r=er,sigma=max(sg*ss[li],1e-3))))
        sar,dT,*_=pennes_peak(f,sp,50.0); sars.append(sar); dTs.append(dT)
    sars=np.array(sars); dTs=np.array(dTs)
    print('%.2f GHz: peakSAR nom=%.3f  range[%.3f,%.3f] (x%.2f)  | dT nom=%.3f  range[%.3f,%.3f] (x%.2f)'%(
        f/1e9, sar0, sars.min(), sars.max(), sars.max()/sars.min(),
        dT0, dTs.min(), dTs.max(), dTs.max()/dTs.min()))
