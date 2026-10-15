import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'serif','font.size':7,'axes.linewidth':0.6})
# Fig 1: protocols schematic
fig,ax=plt.subplots(figsize=(3.45,1.55))
C={'train':'#4C72B0','val':'#DD8452','test':'#55A868','gap':'#BBBBBB','purge':'#FFFFFF'}
import random; random.seed(3)
rows=[('LOACO (grouped)',None),('Natural novelty',None),('Purged block',None)]
n=20
# LOACO: shuffled blocks
lo=['train']*12+['val']*4+['test']*4; random.shuffle(lo)
nn=['train']*12+['gap']+['val']*3+['test']*4
pb=['train','purge','test','purge','train','train','purge','val','purge','train','train','purge','test','purge','train','purge','val','purge','train','train']
for r,(name,seq) in enumerate(zip([x[0] for x in rows],[lo,nn,pb])):
    y=2-r
    for i,s in enumerate(seq):
        ax.add_patch(Rectangle((i,y+0.15),0.92,0.6,facecolor=C[s],edgecolor='#555555' if s=='purge' else 'none',hatch='////' if s=='purge' else None,linewidth=0.4))
    ax.text(-0.4,y+0.45,name,ha='right',va='center')
# annotations
ax.text(10,2.88,'held-out class removed from all train blocks',ha='center',fontsize=6)
ax.text(17.5,1.86,'unseen classes appear here',ha='center',fontsize=6)
ax.annotate('',xy=(20,-0.08),xytext=(0,-0.08),arrowprops=dict(arrowstyle='->',lw=0.6))
ax.text(10,-0.32,'time (1-h blocks; 10-min blocks for purged)',ha='center',fontsize=6)
ax.set_xlim(-7.2,20.3); ax.set_ylim(-0.45,3.05); ax.axis('off')
from matplotlib.patches import Patch
h=[Patch(facecolor=C[k],label=k) for k in ['train','val','test','gap']]+[Patch(facecolor='white',edgecolor='#555555',hatch='////',label='purged')]
ax.legend(handles=h,ncol=5,loc='lower center',bbox_to_anchor=(0.6,-0.2),frameon=False,fontsize=6,handlelength=1.2,columnspacing=0.8)
plt.savefig('fig_protocols.pdf',bbox_inches='tight',pad_inches=0.02)
# Fig 2: D3 shared classes, mean ± seed sd, LOACO vs natural
sd=json.load(open('sd.json'))
dets=['AHSD stress','Mahalanobis','kNN','IF','AE','CLAD']
lab={'AHSD stress':'Stress','Mahalanobis':'Mahal.','kNN':'kNN','IF':'IF','AE':'AE','CLAD':'CLAD'}
cls=['Backdoor','password','xss']; mk={'Backdoor':'o','password':'s','xss':'^'}
fig,ax=plt.subplots(figsize=(3.45,1.75))
for i,d in enumerate(dets):
    for j,(ev,col,off) in enumerate([('LOACO D3','#4C72B0',-0.18),('Natural novelty D3','#C44E52',0.18)]):
        for k,c in enumerate(cls):
            m,s=sd['|'.join([ev,d,c])]
            x=i+off+(k-1)*0.07
            ax.errorbar(x,m,yerr=s,fmt=mk[c],ms=2.6,color=col,elinewidth=0.6,capsize=0,mew=0)
ax.axhline(0.5,color='grey',lw=0.5,ls='--')
ax.set_xticks(range(len(dets))); ax.set_xticklabels([lab[d] for d in dets])
ax.axvline(2.5,color='#999999',lw=0.4)
ax.text(1,0.03,'supervised backbone',ha='center',fontsize=6,color='#444444')
ax.text(4.5,0.03,'benign-only / contrastive',ha='center',fontsize=6,color='#444444')
ax.set_ylabel('AUC (mean $\\pm$ seed s.d.)'); ax.set_ylim(0,1.05)
from matplotlib.lines import Line2D
hh=[Line2D([],[],color='#4C72B0',marker='o',ls='',ms=3,label='LOACO'),Line2D([],[],color='#C44E52',marker='o',ls='',ms=3,label='Natural novelty')]+[Line2D([],[],color='k',marker=mk[c],ls='',ms=3,label=c) for c in cls]
ax.legend(handles=hh,ncol=5,loc='lower center',bbox_to_anchor=(0.5,1.0),frameon=False,fontsize=6,handletextpad=0.2,columnspacing=0.7)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.savefig('fig_d3.pdf',bbox_inches='tight',pad_inches=0.02)
plt.savefig('fig_d3.png',dpi=200,bbox_inches='tight'); 
