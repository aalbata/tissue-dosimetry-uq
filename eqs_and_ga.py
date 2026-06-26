import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'cm'})

# ---------------- equation images (mathtext) ----------------
EQS = {
 'eq_colecole': r"$\varepsilon_r^*(\omega)=\varepsilon_\infty+\sum_{n=1}^{4}\dfrac{\Delta\varepsilon_n}{1+(j\omega\tau_n)^{1-\alpha_n}}+\dfrac{\sigma_i}{j\omega\varepsilon_0}$",
 'eq_effcond': r"$\varepsilon_r^*=\varepsilon_r'-j\varepsilon_r'',\qquad \sigma_{\mathrm{eff}}(\omega)=\omega\varepsilon_0\varepsilon_r''$",
 'eq_gammaeta': r"$\gamma_i=j\omega\sqrt{\mu_0\varepsilon_0\varepsilon_{r,i}^*},\qquad \eta_i=\sqrt{\mu_0/(\varepsilon_0\varepsilon_{r,i}^*)}$",
 'eq_zin': r"$Z_i=\eta_i\,\dfrac{Z_{i+1}+\eta_i\tanh(\gamma_i d_i)}{\eta_i+Z_{i+1}\tanh(\gamma_i d_i)},\quad Z_{N+1}=\eta_{\mathrm{muscle}}$",
 'eq_gamma0': r"$\Gamma_0=\dfrac{Z_1-\eta_0}{Z_1+\eta_0},\qquad E_{\mathrm{inc}}=\sqrt{S_{\mathrm{inc}}\,\eta_0}$",
 'eq_field': r"$E_i(x)=E_i^{+}e^{-\gamma_i x}+E_i^{-}e^{\gamma_i x},\quad H_i(x)=\dfrac{E_i^{+}e^{-\gamma_i x}-E_i^{-}e^{\gamma_i x}}{\eta_i}$",
 'eq_sar': r"$\mathrm{SAR}(x)=\dfrac{\sigma_{\mathrm{eff}}(x)\,|E_{\mathrm{rms}}(x)|^2}{\rho(x)}$",
 'eq_energy': r"$\epsilon_E=\dfrac{|\int_0^L \rho\,\mathrm{SAR}\,dx-(1-|\Gamma_0|^2)S_{\mathrm{inc}}|}{(1-|\Gamma_0|^2)S_{\mathrm{inc}}}$",
 'eq_pennes': r"$\dfrac{d}{dx}[k(x)\dfrac{dT}{dx}]-\rho_b c_b\,\omega_b(x)\,[T-T_a]+Q_{\mathrm{met}}+\rho(x)\,\mathrm{SAR}(x)=0$",
 'eq_bc': r"$-k(0)\dfrac{dT}{dx}|_{0}=h\,[T(0)-T_\infty],\qquad T(L)=T_c$",
 'eq_perf': r"$\omega_b(x)=\dfrac{P(x)\,\rho(x)\times10^{-6}}{60}\ \ [\mathrm{s}^{-1}],\quad k_{m+1/2}=\dfrac{2k_m k_{m+1}}{k_m+k_{m+1}}$",
 'eq_conformal': r"$C(x)=[\hat{\mu}(x)-q,\ \hat{\mu}(x)+q],\qquad \mathrm{Pr}\{Y\in C(X)\}\geq 1-\alpha$",
 'eq_sobol': r"$S_i=\dfrac{\mathrm{Var}_{x_i}(\mathrm{E}_{x_{\sim i}}[Y\,|\,x_i])}{\mathrm{Var}(Y)},\qquad S_{T_i}=\dfrac{\mathrm{E}_{x_{\sim i}}[\mathrm{Var}_{x_i}(Y\,|\,x_{\sim i})]}{\mathrm{Var}(Y)}$",
}
dims={}
for name,tex in EQS.items():
    fig=plt.figure(figsize=(8,1.0)); fig.text(0.01,0.5,tex,fontsize=19,va='center',ha='left')
    fig.savefig(name+'.png',dpi=200,bbox_inches='tight',pad_inches=0.04,transparent=False)
    plt.close(fig)
    from PIL import Image
    w,h=Image.open(name+'.png').size; dims[name]=[w,h]
json.dump(dims, open('eq_dims.json','w'))
print('equations rendered:', len(EQS))

# ---------------- new graphical abstract (carries a result) ----------------
import json as J
R=J.load(open('revision_results.json'))
fig,ax=plt.subplots(figsize=(9.4,2.7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def pbox(x,w,title):
    ax.add_patch(Rectangle((x,0.16),w,0.72,fill=False,lw=1.1,ec='#444'))
    ax.text(x+w/2,0.93,title,ha='center',va='center',fontsize=8.2,weight='bold')
def arr(x0,x1,y=0.52):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=12,lw=1.3,color='#444'))

# A
pbox(0.01,0.215,"A  IT'IS V5.0 inputs")
ax.text(0.025,0.74,"112 tissues",fontsize=7.4)
ax.text(0.025,0.66,r"$\sigma$: conformal interval",fontsize=7.4)
ax.text(0.025,0.59,"thermal: min/max",fontsize=7.4)
for k,yy in enumerate([0.46,0.38,0.30]):
    ax.plot([0.04,0.105],[yy,yy],lw=1.4,color='#3b7dd8'); ax.plot([0.04,0.04],[yy-.012,yy+.012],lw=1,color='#3b7dd8'); ax.plot([0.105,0.105],[yy-.012,yy+.012],lw=1,color='#3b7dd8'); ax.plot(0.0725,yy,'o',ms=2.5,color='#16324f')
ax.text(0.025,0.22,"sampled ranges",fontsize=6.8,style='italic')
arr(0.232,0.272)

# B
pbox(0.28,0.215,"B  Layered RF + bioheat")
ax.add_patch(FancyArrowPatch((0.30,0.74),(0.345,0.74),arrowstyle='-|>',mutation_scale=9,lw=1,color='#888'))
x0=0.355; h=0.34; sw,fw,mw=0.022,0.05,0.06
for xx,ww,c in [(x0,sw,'#f3d7c0'),(x0+sw,fw,'#fdeecb'),(x0+sw+fw,mw,'#f0c9c0')]:
    ax.add_patch(Rectangle((xx,0.36),ww,h,fc=c,ec='#555',lw=0.8))
ax.text(x0+sw/2,0.33,"sk",fontsize=6,ha='center'); ax.text(x0+sw+fw/2,0.33,"fat",fontsize=6,ha='center'); ax.text(x0+sw+fw+mw/2,0.33,"muscle",fontsize=6,ha='center')
ax.text(0.30,0.25,"TL field + Pennes solve",fontsize=7)
arr(0.497,0.537)

# C  propagated bounds (2.45 GHz)
pbox(0.545,0.215,"C  Propagated bounds")
ax.text(0.553,0.79,"2.45 GHz, 50 W/m$^2$",fontsize=7)
sb=R['nominal']['2.45']['SAR_band']; sn=R['nominal']['2.45']['SAR_nom']
tb=R['nominal']['2.45']['dT_band']; tn=R['nominal']['2.45']['dT_nom']
def bar(xc,lo,hi,nom,ymin,ymax,lab,val):
    def nz(v): return 0.30+(v-ymin)/(ymax-ymin)*0.36
    ax.plot([xc,xc],[nz(lo),nz(hi)],lw=5,solid_capstyle='round',color='#3b7dd8' if 'SAR' in lab else '#d8662b')
    ax.plot(xc,nz(nom),'o',ms=4,color='#16324f')
    ax.text(xc,0.70,lab,fontsize=6.8,ha='center'); ax.text(xc,0.235,val,fontsize=6.2,ha='center')
bar(0.60,sb[0],sb[1],sn,0,10,"peak SAR",("%.1f\n[%.1f,%.1f]"%(sn,sb[0],sb[1])))
bar(0.69,tb[0],tb[1],tn,0,0.8,r"peak $\Delta T$",("%.2f\n[%.2f,%.2f]"%(tn,tb[0],tb[1])))
arr(0.762,0.802)

# D dominant driver
pbox(0.81,0.18,"D  Dominant driver")
ax.text(0.82,0.74,"SAR:",fontsize=7,weight='bold'); ax.text(0.90,0.74,r"$\sigma$ (all f)",fontsize=7.2,ha='center')
ax.text(0.82,0.58,r"$\Delta T$:",fontsize=7,weight='bold')
ax.text(0.825,0.47,"0.9 GHz",fontsize=6,); ax.text(0.95,0.47,r"perfusion",fontsize=6.6,ha='right')
ax.text(0.825,0.39,"5.8 GHz",fontsize=6,); ax.text(0.95,0.39,r"$\sigma$",fontsize=7.4,ha='right')
ax.annotate("",xy=(0.93,0.41),xytext=(0.86,0.49),arrowprops=dict(arrowstyle='->',lw=0.9,color='#666'))
ax.text(0.815,0.27,"driver shifts\nwith frequency",fontsize=6.4,style='italic')

ax.text(0.5,0.045,"Database-level property uncertainty changes SAR and temperature margins; the controlling property depends on endpoint and frequency.",
        ha='center',fontsize=8.2,weight='bold')
fig.savefig('fig_graphabs.pdf',bbox_inches='tight')
print('graphical abstract rebuilt')
