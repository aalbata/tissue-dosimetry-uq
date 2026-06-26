import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
EPS0 = 8.8541878128e-12

def load_table(f='Thermal_dielectric_acoustic_MR properties_database_V5.0(Excel).xls'):
    d=pd.read_excel(f, sheet_name='Sheet1', header=None)
    grp=d.iloc[1].ffill(); sub=d.iloc[2]
    cols=[]
    for g,s in zip(grp,sub):
        g=('' if pd.isna(g) else str(g).strip()); s=('' if pd.isna(s) else str(s).strip())
        cols.append((g+' | '+s).strip(' |'))
    data=d.iloc[3:].copy(); data.columns=cols; data=data.reset_index(drop=True)
    data=data.rename(columns={data.columns[1]:'Tissue'})
    return data, cols

def cole_cole_params(row):
    g=lambda i: pd.to_numeric(row.iloc[i], errors='coerce')
    ef=g(27)
    de=[g(28),g(31),g(35),g(38)]
    tau=[g(29)*1e-12, g(32)*1e-9, g(36)*1e-6, g(39)*1e-3]  # ps,ns,us,ms -> s
    alf=[g(30),g(33),g(37),g(40)]
    sig=g(34)
    return ef,de,tau,alf,sig

def epsilon_complex(freq_hz, ef, de, tau, alf, sig):
    w=2*np.pi*freq_hz
    eps = complex(ef)
    for dn,tn,an in zip(de,tau,alf):
        if np.isnan(dn) or np.isnan(tn): continue
        an = 0.0 if np.isnan(an) else an
        eps += dn/(1+(1j*w*tn)**(1-an))
    if not np.isnan(sig):
        eps += sig/(1j*w*EPS0)
    return eps  # complex relative permittivity eps' - j eps''

def eps_sigma(freq_hz, params):
    ef,de,tau,alf,sig=params
    e=epsilon_complex(freq_hz,ef,de,tau,alf,sig)
    eps_r=e.real
    eps_im=-e.imag
    sigma=w_sigma=2*np.pi*freq_hz*EPS0*eps_im
    return eps_r, sigma

if __name__=='__main__':
    data,cols=load_table()
    names=data['Tissue'].astype(str)
    # reference values (IT'IS/Gabriel model), approx
    ref={
      'Muscle': {0.9e9:(55.0,0.943), 2.45e9:(52.7,1.74)},
      'Fat':    {0.9e9:(5.46,0.0511),2.45e9:(5.28,0.105)},
      'Skin (Dry)':{0.9e9:(41.4,0.867),2.45e9:(38.0,1.46)},
    }
    for tname in ref:
        idx=names[names.str.fullmatch(tname, case=False)].index
        if len(idx)==0:
            idx=names[names.str.contains(tname.split()[0], case=False, na=False)].index
        i=idx[0]
        p=cole_cole_params(data.iloc[i])
        print('=== %s (row %d: %s)'%(tname,i,names.iloc[i]))
        for fr,(re_ref,sg_ref) in ref[tname].items():
            er,sg=eps_sigma(fr,p)
            print('  %4.2f GHz  eps_r=%6.2f (ref %6.2f)  sigma=%6.3f (ref %6.3f)'%(fr/1e9,er,re_ref,sg,sg_ref))
