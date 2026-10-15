import re, json
L = open('/mnt/user-data/uploads/RESULTS_SUMMARY.md').read().splitlines()
evals = {}; cur=None; i=0; transfer=[]
sec=None
while i < len(L):
    l=L[i]
    if l.startswith('## '): sec=l
    m=re.match(r'^\*\*(.+?)\*\* \((\d+) classes?: (.+)\)$', l)
    if m: cur=m.group(1)
    if l.startswith('| Detector | family |') and cur and sec and (sec.startswith('## R1') or sec.startswith('## R2') or sec.startswith('## R3')):
        hdr=[c.strip() for c in l.strip('|').split('|')]
        classes=hdr[2:-1]; rows=[]
        i+=2
        while i<len(L) and L[i].startswith('|'):
            c=[x.strip() for x in L[i].strip('|').split('|')]
            vals=[float(x.split()[0]) for x in c[2:-1]]
            rows.append((c[0],c[1],vals,float(c[-1])))
            i+=1
        evals[cur]=dict(classes=classes,rows=rows); continue
    if l.startswith('| Detector | source |'):
        i+=2
        while i<len(L) and L[i].startswith('|'):
            c=[x.strip() for x in L[i].strip('|').split('|')]
            transfer.append((c[0],c[1],[float(x.split()[0]) for x in c[2:]]))
            i+=1
        continue
    i+=1
json.dump(dict(evals=evals,transfer=transfer),open('data.json','w'),indent=1)
for k,v in evals.items(): print(k, v['classes'], len(v['rows']))
print(len(transfer))
