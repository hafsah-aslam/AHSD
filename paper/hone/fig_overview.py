import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
import random
plt.rcParams.update({'font.family':'serif','font.size':6.5})
fig=plt.figure(figsize=(7.16,2.05))
W=[0.235,0.27,0.235,0.235]; gap=0.008; x0=0.0
axes=[]
for w in W:
    ax=fig.add_axes([x0,0.0,w,0.9]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); axes.append(ax); x0+=w+gap
titles=['(a) Leakage-controlled data','(b) Paired zero-day protocols','(c) 12 detectors, 5 families','(d) Evaluation and acceptance']
for ax,t in zip(axes,titles):
    ax.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98,boxstyle='round,pad=0,rounding_size=0.03',fc='#F7F7F7',ec='#888888',lw=0.6))
    ax.text(0.5,1.04,t,ha='center',va='bottom',fontsize=7,fontweight='bold')
def chain(ax,items,fc):
    n=len(items); h=0.115; top=0.92; step=(0.86)/n
    for i,it in enumerate(items):
        y=top-i*step-h
        ax.add_patch(FancyBboxPatch((0.08,y),0.84,h,boxstyle='round,pad=0,rounding_size=0.02',fc=fc,ec='#555555',lw=0.5))
        ax.text(0.5,y+h/2,it,ha='center',va='center',fontsize=6.2)
        if i<n-1: ax.annotate('',xy=(0.5,y-step+h+0.005),xytext=(0.5,y-0.005),arrowprops=dict(arrowstyle='->',lw=0.6))
chain(axes[0],['NetFlow-v3 D1–D4 + Lycos2017','Drop IDs; train-only scaling','1-h time blocks (10-min, D5)','Windows: T=32, stride 16','SHA-256 deduplication','Fail-closed integrity checks'],'#DCE6F2')
# (b) protocol strips
ax=axes[1]
C={'train':'#4C72B0','val':'#DD8452','test':'#55A868','gap':'#BBBBBB'}
random.seed(3)
lo=['train']*12+['val']*4+['test']*4; random.shuffle(lo)
nn=['train']*12+['gap']+['val']*3+['test']*4
rc=['train']*7+['gap']+['test']*12
pb=['train','p','test','p','train','train','p','val','p','train','train','p','test','p','train','p','val','p','train','train']
rows=[('LOACO',lo,'one class removed from all train blocks'),('Natural novelty',nn,'classes first seen after training'),('Rolling cut-points',rc,'cut moved to 30/45/60/75% of the data'),('Purged block',pb,'blocks next to test/val dropped')]
for r,(name,seq,note) in enumerate(rows):
    y=0.80-r*0.205
    ax.text(0.05,y+0.075,name,fontsize=6.4,fontweight='bold',va='bottom')
    for i,s in enumerate(seq):
        x=0.05+i*0.045
        if s=='p': ax.add_patch(Rectangle((x,y),0.04,0.06,fc='white',ec='#555555',hatch='////',lw=0.3))
        else: ax.add_patch(Rectangle((x,y),0.04,0.06,fc=C[s],ec='none'))
    ax.text(0.05,y-0.035,note,fontsize=5.6,style='italic',va='center',color='#333333')
ax.annotate('',xy=(0.95,0.045),xytext=(0.05,0.045),arrowprops=dict(arrowstyle='->',lw=0.6)); ax.text(0.5,0.05,'time',ha='center',va='bottom',fontsize=5.6)
h=[Patch(fc=C[k],label=k) for k in C]+[Patch(fc='white',ec='#555555',hatch='////',label='purged')]
ax.legend(handles=h,ncol=5,loc='upper center',bbox_to_anchor=(0.5,0.035),frameon=False,fontsize=5.4,handlelength=0.9,columnspacing=0.5,handletextpad=0.3)
# (c) families
ax=axes[2]
fams=[('Confidence ×2','MSP, Energy, P(attack)$^\\dagger$','#F2DCDB'),('Representation ×2','Stress, Mahalanobis, kNN','#E4DFEC'),('Benign-only','IF, OCSVM, PCA, AE','#EBF1DE'),('Contrastive','CLAD','#FDE9D9'),('Reference','XGBoost, XGBoost-s','#DCE6F2')]
for i,(f,d,c) in enumerate(fams):
    y=0.80-i*0.172
    ax.add_patch(FancyBboxPatch((0.06,y),0.88,0.14,boxstyle='round,pad=0,rounding_size=0.02',fc=c,ec='#555555',lw=0.5))
    ax.text(0.5,y+0.095,f,ha='center',va='center',fontsize=6.4,fontweight='bold'); ax.text(0.5,y+0.04,d,ha='center',va='center',fontsize=6)
# (d)
chain(axes[3],['5 seeds per configuration','AUC; TPR at 1% FPR; PR-AUC','Seed s.d.: stability','Family ranking per protocol','Checkpoint sensitivity','Acceptance protocol (8 checks)'],'#EBF1DE')
# arrows between panels
for i in range(3):
    xa=sum(W[:i+1])+gap*i
    fig.patches.append(matplotlib.patches.FancyArrowPatch((xa-0.004,0.45),(xa+gap+0.004,0.45),transform=fig.transFigure,arrowstyle='->',mutation_scale=7,lw=0.8,color='#333333'))
plt.savefig('fig_overview.pdf',bbox_inches='tight',pad_inches=0.02); plt.savefig('fig_overview.png',dpi=170,bbox_inches='tight',pad_inches=0.02)
