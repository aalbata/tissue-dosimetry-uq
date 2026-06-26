import numpy as np, pandas as pd
from cole_cole import load_table, cole_cole_params, eps_sigma
from dosimetry import slab_SAR

RHO_B=1050.0; C_B=3617.0; T_ART=37.0; T_CORE=37.0; T_AIR=24.0; H_CONV=8.0

def prop_row(data, names, name):
    return names[names.str.fullmatch(name, case=False)].index[0]

def get_props(data, i, freq, which='avg'):
    g=lambda c: pd.to_numeric(data.iloc[i,c], errors='coerce')
    # which: 'avg','min','max' for the bounded thermal props
    cmap={'rho':(2,5,6),'c':(7,10,11),'k':(12,15,16),'perf':(17,20,21),'met':(22,25,26)}
    sel={'avg':0,'min':1,'max':2}[which]
    def val(key):
        a=g(cmap[key][0]); 
        if which=='avg' or np.isnan(g(cmap[key][1+ (0 if sel==1 else 1) ])):
            pass
        v=g(cmap[key][sel]) if which!='avg' else a
        return a if (np.isnan(v)) else v
    rho=val('rho'); c=val('c'); k=val('k'); perf=val('perf'); met=val('met')
    er,sg=eps_sigma(freq, cole_cole_params(data.iloc[i]))
    return dict(rho=rho,c=c,k=k,perf=perf,met=(0.0 if np.isnan(met) else met),eps_r=er,sigma=sg)

def solve_pennes(freq, layer_specs, S_inc=50.0, L_tail=0.05, N=1200):
    """layer_specs: list of (thickness or None, props_dict). Last is semi-infinite (use L_tail)."""
    ds=[]; 
    for t,_ in layer_specs[:-1]: ds.append(t)
    ds.append(L_tail)
    Ltot=sum(ds)
    x=np.linspace(0,Ltot,N); dx=x[1]-x[0]
    # assign per-node properties
    edges=[0.0]
    for d in ds: edges.append(edges[-1]+d)
    idx_layer=np.zeros(N,dtype=int)
    for li in range(len(ds)):
        idx_layer[(x>=edges[li])&(x<=edges[li+1])]=li
    P=[lp[1] for lp in layer_specs]
    k=np.array([P[idx_layer[i]]['k'] for i in range(N)])
    rho=np.array([P[idx_layer[i]]['rho'] for i in range(N)])
    perf=np.array([P[idx_layer[i]]['perf'] for i in range(N)])
    met=np.array([P[idx_layer[i]]['met'] for i in range(N)])
    wb=perf*rho*1e-6/60.0                     # 1/s volumetric perfusion
    Wb=RHO_B*C_B*wb                           # W/m^3/K sink coefficient
    # SAR profile on this grid
    layers_em=[dict(d=(ds[li] if li<len(ds)-1 else None),
                    eps_r=P[li]['eps_r'],sigma=P[li]['sigma'],rho=P[li]['rho']) for li in range(len(ds))]
    xs,sar,Gamma=slab_SAR(freq, layers_em, S_inc=S_inc)
    SAR=np.interp(x, xs, sar, left=sar[0], right=sar[-1])
    def assemble(include_sar):
        A=np.zeros((N,N)); b=np.zeros(N)
        # surface Robin: k*(T1-T0)/dx = H_CONV*(T0 - T_AIR)
        A[0,0]=k[0]/dx + H_CONV; A[0,1]=-k[0]/dx; b[0]=H_CONV*T_AIR
        for i in range(1,N-1):
            A[i,i-1]=k[i]/dx**2
            A[i,i+1]=k[i]/dx**2
            A[i,i]=-2*k[i]/dx**2 - Wb[i]
            q=rho[i]*met[i] + (rho[i]*SAR[i] if include_sar else 0.0)
            b[i]=-Wb[i]*T_ART - q
        A[N-1,N-1]=1.0; b[N-1]=T_CORE
        return np.linalg.solve(A,b)
    T_exp=assemble(True); T_base=assemble(False)
    return x, T_exp, T_base, SAR, Gamma

if __name__=='__main__':
    data,cols=load_table(); names=data['Tissue'].astype(str)
    f=2.45e9
    sk=prop_row(data,names,'Skin'); ft=prop_row(data,names,'Fat'); ms=prop_row(data,names,'Muscle')
    specs=[(0.0015,get_props(data,sk,f)),(0.005,get_props(data,ft,f)),(None,get_props(data,ms,f))]
    x,Te,Tb,SAR,G=solve_pennes(f,specs,S_inc=50.0)
    dT=Te-Tb
    print('f=2.45 GHz, S_inc=50 W/m^2')
    print('  |Gamma|=%.3f  peak SAR=%.3f W/kg'%(abs(G),SAR.max()))
    print('  baseline skin temp=%.2f C, exposed skin temp=%.2f C'%(Tb[0],Te[0]))
    print('  peak dT=%.4f C at depth %.2f mm'%(dT.max(), x[dT.argmax()]*1000))
    print('  dT at surface=%.4f C'%dT[0])
