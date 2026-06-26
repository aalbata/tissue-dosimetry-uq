import numpy as np
from cole_cole import load_table, cole_cole_params, eps_sigma, EPS0
MU0=4e-7*np.pi; C0=299792458.0; ETA0=np.sqrt(MU0/EPS0)

def layer_em(freq, eps_r, sigma):
    w=2*np.pi*freq
    epc=EPS0*eps_r - 1j*sigma/w           # complex permittivity
    gamma=1j*w*np.sqrt(MU0*epc)           # propagation constant (Re=alpha attenuation)
    eta=np.sqrt(1j*w*MU0/(sigma+1j*w*EPS0*eps_r))
    return gamma, eta

def slab_SAR(freq, layers, S_inc=10.0, nx=2000, tail_len=0.05):
    """layers: list of dicts {d(m or None for semi-inf), eps_r, sigma, rho}. Plane wave, normal incidence."""
    w=2*np.pi*freq
    # intrinsic impedance + gamma per layer (air first, then tissue layers; last semi-infinite)
    g=[ (1j*w/C0, ETA0) ]  # air
    for L in layers:
        g.append(layer_em(freq, L['eps_r'], L['sigma']))
    # input impedance from back to front (last layer semi-infinite => Zin=eta_last)
    N=len(layers)
    eta=[g[i][1] for i in range(N+1)]
    gam=[g[i][0] for i in range(N+1)]
    Zin=[None]*(N+1)
    Zin[N]=eta[N]
    for i in range(N-1,0,-1):
        d=layers[i-1]['d']
        gd=gam[i]*d
        Zin[i]=eta[i]*(Zin[i+1]+eta[i]*np.tanh(gd))/(eta[i]+Zin[i+1]*np.tanh(gd))
    # reflection at air/first interface
    Gamma=(Zin[1]-eta[0])/(Zin[1]+eta[0])
    Einc=np.sqrt(S_inc*ETA0)              # RMS E in air (S=E_rms^2/eta0)
    # forward/backward in each layer via chain; track E at interfaces
    # E_total at air side of interface 1:
    Ep=[None]*(N+1); Em=[None]*(N+1)      # forward/backward amplitudes per layer at its front face
    # air
    Ep[0]=Einc; Em[0]=Gamma*Einc
    # continuity: at each interface match E and H. Use transfer via reflection coeff at each interface.
    # Compute layer-by-layer forward amplitude using interface reflection from Zin.
    # E field just inside layer i front face:
    E_if=Einc*(1+Gamma)                   # total E at interface 1 (continuous)
    # Propagate through stack computing |E|(x) and SAR
    xs=[]; sar=[]; 
    # We march layer by layer. At front of layer i we know total E (E_front) and total H (E_front/Zin[i+1... )
    # Simpler: reconstruct forward/backward from total E and H at front face.
    Efront=E_if
    Hfront=Efront/Zin[1]                   # H = E/Zin at interface (looking into stack)
    x0=0.0
    for i in range(1,N+1):
        et=eta[i]; gm=gam[i]
        # forward/backward amplitudes at front face: E=A+B, H=(A-B)/eta
        A=0.5*(Efront+Hfront*et)
        B=0.5*(Efront-Hfront*et)
        if i<N:
            d=layers[i-1]['d']
        else:
            d=tail_len  # semi-infinite layer integration depth
        npts=max(50,int(nx*d/0.02))
        xx=np.linspace(0,d,npts)
        Ex=A*np.exp(-gm*xx)+B*np.exp(gm*xx)
        sg=layers[i-1]['sigma']; rho=layers[i-1]['rho']
        s=sg*np.abs(Ex)**2/rho             # SAR (W/kg), Ex is RMS
        xs.append(x0+xx); sar.append(s)
        # update front face for next layer: fields at end of this layer
        Eend=A*np.exp(-gm*d)+B*np.exp(gm*d)
        Hend=(A*np.exp(-gm*d)-B*np.exp(gm*d))/et
        Efront=Eend; Hfront=Hend; x0=x0+d
    xs=np.concatenate(xs); sar=np.concatenate(sar)
    # energy check: absorbed power per area = integral sigma|E|^2 dx (=rho*SAR) ; should = (1-|Gamma|^2)*S_inc
    return xs, sar, Gamma

def absorbed_check(freq, layers, S_inc=10.0):
    xs,sar,Gamma=slab_SAR(freq,layers,S_inc)
    # recompute rho along x to integrate volumetric power
    # rebuild rho profile
    bounds=[]; acc=0
    rho_prof=np.zeros_like(xs)
    # assign rho by depth
    edges=[0]; 
    for L in layers[:-1]: edges.append(edges[-1]+L['d'])
    edges.append(edges[-1]+0.05)
    for i,L in enumerate(layers):
        m=(xs>=edges[i])&(xs<=edges[i+1])
        rho_prof[m]=L['rho']
    Pvol=np.trapezoid(rho_prof*sar, xs)        # W/m^2 absorbed
    Pin=(1-np.abs(Gamma)**2)*S_inc
    return Pvol, Pin, Gamma

if __name__=='__main__':
    data,cols=load_table()
    names=data['Tissue'].astype(str)
    def prop(name, f):
        i=names[names.str.fullmatch(name,case=False)].index[0]
        er,sg=eps_sigma(f, cole_cole_params(data.iloc[i]))
        rho=float(__import__('pandas').to_numeric(data.iloc[i,2],errors='coerce'))
        return er,sg,rho,i
    for f in [0.9e9, 2.45e9]:
        sk=prop('Skin',f); ft=prop('Fat',f); ms=prop('Muscle',f)
        layers=[dict(d=0.0015,eps_r=sk[0],sigma=sk[1],rho=sk[2]),
                dict(d=0.005, eps_r=ft[0],sigma=ft[1],rho=ft[2]),
                dict(d=None,  eps_r=ms[0],sigma=ms[1],rho=ms[2])]
        Pvol,Pin,Gamma=absorbed_check(f,layers,S_inc=10.0)
        xs,sar,_=slab_SAR(f,layers,S_inc=10.0)
        print('f=%.2f GHz  |Gamma|=%.3f  Pabs(integ)=%.4f W/m2  (1-R)Sinc=%.4f W/m2  ratio=%.3f  peakSAR=%.4f W/kg'
              %(f/1e9,abs(Gamma),Pvol,Pin,Pvol/Pin,sar.max()))
