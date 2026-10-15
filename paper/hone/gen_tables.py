import json
d=json.load(open('data.json'))
abbr={'P(attack)†':'P(att.)$^\\dagger$','AHSD stress':'Stress','Mahalanobis':'Mahal.','Isolation Forest':'IF'}
cls_abbr={'Infilteration':'Infil.','Infiltration':'Infil.','Web Attacks':'Web','BruteForce':'Brute F.','Reconnaissance':'Recon.','Shellcode':'Shellc.','ransomware':'ransom.','dos_hulk':'DoS Hulk','Analysis':'Anal.','Backdoor':'Backd.','Exploits':'Expl.','Fuzzers':'Fuzz.','Generic':'Gen.','injection':'inject.','password':'passw.','scanning':'scan.','portscan':'portscan'}
def tab(name,label,cap):
    e=d['evals'][name]; cl=e['classes']
    s=['\\begin{table}[!t]','\\centering','\\scriptsize','\\setlength{\\tabcolsep}{2.6pt}',
       '\\caption{%s}'%cap,'\\label{%s}'%label,
       '\\begin{tabular}{l'+'c'*len(cl)+'c}','\\toprule',
       'Detector & '+' & '.join(cls_abbr.get(c,c) for c in cl)+' & s.d.\\\\','\\midrule']
    prev=None
    for det,fam,vals,sd in e['rows']:
        if prev and fam!=prev: s.append('\\midrule')
        prev=fam
        s.append(abbr.get(det,det)+' & '+' & '.join('%.2f'%v for v in vals)+' & %.3f\\\\'%sd)
    s+=['\\bottomrule','\\end{tabular}','\\end{table}']
    return '\n'.join(s)
out=[]
specs=[('LOACO D1 (families)','tab:a_l1','Per-class AUC, LOACO D1 (CIC-IDS2018 families). Mean of five seeds; s.d.\\ = mean seed standard deviation over classes.'),
 ('LOACO D2','tab:a_l2','Per-class AUC, LOACO D2 (UNSW-NB15).'),
 ('LOACO D3','tab:a_l3','Per-class AUC, LOACO D3 (ToN-IoT).'),
 ('Natural novelty D1','tab:a_n1','Per-class AUC, natural novelty D1.'),
 ('Natural novelty D3','tab:a_n3','Per-class AUC, natural novelty D3.'),
 ('Natural novelty D5','tab:a_n5','Per-class AUC, natural novelty D5 (Lycos2017).'),
 ('LOACO D5 purged_block','tab:a_p5','AUC, purged-block LOACO D5 (single fold).')]
for n,l,c in specs: out.append(tab(n,l,c))
# transfer
T=d['transfer']; dets=[]
for det,src,v in T:
    if det not in dets: dets.append(det)
s=['\\begin{table*}[!t]','\\centering','\\scriptsize','\\setlength{\\tabcolsep}{3pt}',
   '\\caption{Zero-shot transfer AUC (mean of five seeds). Rows: detector; column groups: source dataset; columns: target. Diagonal cells (in-domain) in grey.}','\\label{tab:a_transfer}',
   '\\begin{tabular}{l'+'cccc'*3+'}','\\toprule',
   ' & \\multicolumn{4}{c}{Source D1} & \\multicolumn{4}{c}{Source D2} & \\multicolumn{4}{c}{Source D3}\\\\',
   '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}',
   'Detector'+' & D1 & D2 & D3 & D4'*3+'\\\\','\\midrule']
for det in dets:
    cells=[]
    for src in ['D1','D2','D3']:
        v=[x for x in T if x[0]==det and x[1]==src][0][2]
        for j,t in enumerate(['D1','D2','D3','D4']):
            c='%.2f'%v[j]
            cells.append('\\cellcolor{black!10}'+c if t==src else c)
    s.append(abbr.get(det,det)+' & '+' & '.join(cells)+'\\\\')
s+=['\\bottomrule','\\end{tabular}','\\end{table*}']
out.append('\n'.join(s))
open('appendix_tables.tex','w').write('\n\n'.join(out))
print(open('appendix_tables.tex').read()[:1500])
