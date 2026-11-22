"""Bound presentation only. Execution always consults the full parent graph."""
import copy
import hashlib
import json


def bounded(value, *, max_chars=20000, max_items=24, max_string=1200):
    original=json.dumps(value,ensure_ascii=False,default=str)
    if len(original)<=max_chars:return copy.deepcopy(value), {'original_chars':len(original),'chars':len(original),'truncated':False}
    omitted=[]
    def trim(x,path,items,strings,depth=0):
        if depth>10:
            omitted.append(path);return {'omitted':'depth_limit'}
        if isinstance(x,str) and len(x)>strings:
            omitted.append(path);return x[:strings]+'… [truncated; full value archived]'
        if isinstance(x,list):
            if len(x)>items:omitted.append(path)
            return [trim(v,f'{path}[{i}]',items,strings,depth+1) for i,v in enumerate(x[:items])]
        if isinstance(x,dict):return {k:trim(v,path+'.'+k,items,strings,depth+1) for k,v in x.items()}
        return x
    items,strings=max_items,max_string
    while True:
        out=trim(value,'$',items,strings);size=len(json.dumps(out,ensure_ascii=False,default=str))
        if size<=max_chars:break
        if items==1 and strings<=32:
            out={'context_omitted':True,'reason':'oversize_diagnostic','automatic_permission':False,
                 'instruction':'Acquire current evidence through documented APIs; omitted history cannot authorize any action.'}
            break
        items=max(1,items//2);strings=max(32,strings//2)
    return out, {'original_chars':len(original),'chars':len(json.dumps(out,ensure_ascii=False)),
        'truncated':True,'omitted_paths':list(dict.fromkeys(omitted))[:24],
        'reason':'presentation_budget_only_full_graph_retained','sha256':hashlib.sha256(original.encode()).hexdigest()}
