"""Original, offline K1 -> K2 NCS ABI adapter. No installation or asset writes.

Raw operands survive unchanged, except ACTION operands and relocated PC-relative
addresses. Runtime behavior, 2DA identity, and native handler semantics are separate
gates. This module deliberately does not use PyKotor's compiler or serializer.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct


class Refused(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


SIZE = {'void': 0, 'action': 0, 'vector': 12, 'int': 4, 'float': 4,
        'string': 4, 'object': 4, 'effect': 4, 'event': 4, 'location': 4, 'talent': 4}
QUAL = {'int': 3, 'float': 4, 'string': 5, 'object': 6,
        'effect': 16, 'event': 17, 'location': 18, 'talent': 19}
JUMPS = {0x1d, 0x1e, 0x1f, 0x25}


def normalized_type(t):
    return {'object_id': 'object'}.get(t.lower(), t.lower())


def strip_comments(text):
    # Keep comment markers inside strings. All retail prototypes are declarations.
    token = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\r\n]*', re.S)
    return token.sub(lambda m: m[0] if m[0].startswith('"') else ' ', text)


def eval_literal(expr, constants, trail=()):
    expr = re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[fF]\b', r'\1', expr.strip())
    ops = {ast.Add: lambda a,b:a+b, ast.Sub:lambda a,b:a-b,
           ast.Mult:lambda a,b:a*b, ast.Div:lambda a,b:a/b,
           ast.BitOr:lambda a,b:a|b, ast.BitAnd:lambda a,b:a&b,
           ast.LShift:lambda a,b:a<<b, ast.RShift:lambda a,b:a>>b}
    def visit(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str,int,float)):
            return n.value
        if isinstance(n, (ast.List,ast.Tuple)):
            return [visit(x) for x in n.elts]
        if isinstance(n, ast.Name):
            if n.id in trail or n.id not in constants:
                raise Refused('Unresolved default/constant: '+n.id)
            return eval_literal(constants[n.id]['expression'], constants, trail+(n.id,))
        if isinstance(n, ast.UnaryOp):
            v=visit(n.operand)
            if isinstance(n.op, ast.USub): return -v
            if isinstance(n.op, ast.UAdd): return v
            if isinstance(n.op, ast.Invert): return ~v
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](visit(n.left),visit(n.right))
        raise Refused('Unsupported default expression: '+expr)
    try:
        return visit(ast.parse(expr,mode='eval').body)
    except (SyntaxError,TypeError) as e:
        raise Refused('Unqualified literal: '+expr) from e


def parse_nwscript(data):
    text=strip_comments(data.decode('cp1252'))
    constants={'OBJECT_SELF':{'type':'object','expression':'0'},
               'OBJECT_INVALID':{'type':'object','expression':'1'}}
    for typ,name,expr in re.findall(r'^\s*(int|float|string|object)\s+(\w+)\s*=\s*([^;]+);',text,re.M):
        constants[name]={'type':typ,'expression':expr.strip()}
    funcs=[]
    pattern=r'^\s*(void|int|float|string|object|vector|effect|event|location|talent)\s+(\w+)\s*\(([^;{}]*)\)\s*;'
    for ret,name,params in re.findall(pattern,text,re.M):
        pp=[]
        chunks=re.findall(r'(?:"(?:\\.|[^"\\])*"|\[[^\]]*\]|[^,"\[])+',params)
        for p in chunks:
            if not p.strip(): continue
            m=re.fullmatch(r'\s*(\w+)\s+(\w+)\s*(?:=\s*(.*))?',p,re.S)
            if not m: raise Refused('Unparsed parameter: '+p)
            typ,pn,default=m.groups();typ=normalized_type(typ)
            if typ not in SIZE: raise Refused('Unknown parameter type: '+typ)
            row={'type':typ,'name':pn,'default_expression':default.strip() if default is not None else None}
            if default is not None:row['default']=eval_literal(default,constants)
            pp.append(row)
        funcs.append({'id':len(funcs),'name':name,'return':ret,'params':pp})
    if not funcs or funcs[0]['name']!='Random' or len({f['name'] for f in funcs})!=len(funcs):
        raise Refused('Unexpected nwscript routine order/duplicate')
    return {'sha256':sha(data),'functions':funcs,'constants':constants}


def make_abi(k1_nss,k2_nss):
    a,b=parse_nwscript(k1_nss),parse_nwscript(k2_nss)
    if len(a['functions'])!=772 or len(b['functions'])!=877:
        raise Refused('Unqualified retail routine counts')
    return {'schema':1,'k1':a,'k2':b,'provenance':'Retail nwscript.nss, read-only; metadata only'}


def load_abi(path=None):
    if path is None:raise Refused('Supply abi=make_abi(owned_k1_nss, owned_k2_nss); no game definitions are bundled')
    return json.loads(Path(path).read_text(encoding='utf8'))


@dataclass(frozen=True)
class Instruction:
    pc: int
    op: int
    qual: int
    raw: bytes
    args: tuple=()
    target: int|None=None

    @property
    def end(self):return self.pc+len(self.raw)


def decode(data):
    if len(data)<15 or data[:9]!=b'NCS V1.0B' or struct.unpack_from('>I',data,9)[0]!=len(data):
        raise Refused('NCS header/size mismatch (trailing data is refused)')
    out=[];pc=13
    scalar={3,4,5,6,16,17,18,19}
    pair={0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x30,0x31,0x32,0x33,0x3a,0x3b,0x3c}
    while pc<len(data):
        if pc+2>len(data):raise Refused('Truncated instruction')
        op,q=data[pc:pc+2];n=2;args=();target=None
        if op==0x2d:raise Refused(f'Forbidden opcode 0x2D at {pc}')
        if op in (1,3,0x26,0x27):
            if q!=1:raise Refused(f'Bad copy qualifier at {pc}')
            n=8
        elif op==2:
            if q not in scalar:raise Refused(f'Bad RSADD qualifier at {pc}')
        elif op==4:
            if q not in (3,4,5,6):raise Refused(f'Bad CONST qualifier at {pc}')
            n=6
            if q==5:
                if pc+4>len(data):raise Refused('Truncated string length')
                n=4+struct.unpack_from('>H',data,pc+2)[0]
        elif op==5:
            if q:raise Refused(f'Bad ACTION qualifier at {pc}')
            n=5
        elif 6<=op<=0x18:
            if q not in pair:raise Refused(f'Bad binary qualifier at {pc}')
            if op in (0xb,0xc) and q==0x24:n=4
        elif op in (0x19,0x1a,0x22):
            if q not in ({3,4} if op==0x19 else {3}):raise Refused(f'Bad unary qualifier at {pc}')
        elif op in ({0x1b}|JUMPS):
            if q:raise Refused(f'Bad branch/MOVSP qualifier at {pc}')
            n=6
        elif op==0x20 or op in (0x2a,0x2b):
            if q:raise Refused(f'Bad control qualifier at {pc}')
        elif op==0x21:
            if q!=1:raise Refused(f'Bad DESTRUCT qualifier at {pc}')
            n=8
        elif op in (0x23,0x24,0x28,0x29):
            if q!=3:raise Refused(f'Bad INC/DEC qualifier at {pc}')
            n=6
        elif op==0x2c:n=10
        else:raise Refused(f'Unknown/unsupported opcode {op:#x} at {pc}')
        if pc+n>len(data):raise Refused(f'Truncated operands at {pc}')
        raw=data[pc:pc+n]
        if op in (1,3,0x26,0x27):args=struct.unpack_from('>iH',raw,2)
        elif op==4:
            args=(raw[4:] if q==5 else struct.unpack_from('>f' if q==4 else '>i',raw,2)[0],)
        elif op==5:args=struct.unpack_from('>HB',raw,2)
        elif op==0x2c:
            args=struct.unpack_from('>II',raw,2);target=pc+q
        elif op==0x21:args=struct.unpack_from('>HHH',raw,2)
        elif n==6:args=struct.unpack_from('>i',raw,2)
        elif n==4:args=struct.unpack_from('>H',raw,2)
        if op in JUMPS:target=pc+args[0]
        out.append(Instruction(pc,op,q,raw,args,target));pc+=n
    starts={i.pc for i in out}
    for i in out:
        if i.target is not None and i.target not in starts:
            raise Refused(f'Jump/state target {i.target} from {i.pc} is not an instruction boundary')
    return out


def call_signature(i,funcs):
    rid,argc=i.args
    if rid>=len(funcs):raise Refused(f'ACTION routine {rid} outside ABI at {i.pc}')
    f=funcs[rid];p=f['params']
    minimum=max((n+1 for n,x in enumerate(p) if x['default_expression'] is None),default=0)
    if not minimum<=argc<=len(p):
        raise Refused(f'ACTION {f["name"]} argc={argc}, expected {minimum}..{len(p)} at {i.pc}')
    return f,p[:argc]


def proven_correlated_edges(code):
    """Prove only adjacent duplicated-condition branches with unique entry.

    No data or bytecode changes. If either test can be entered independently,
    or the condition could be written between tests, leave both edges feasible.
    """
    incoming={i.pc:set() for i in code}
    for i in code:
        successors=[] if i.op==0x20 else [i.end]
        if i.op==0x1d:successors=[i.target]
        elif i.target is not None:successors.append(i.target)
        for target in successors:
            if target in incoming:incoming[target].add(i.pc)
    result={}
    for a,b,c,d in zip(code,code[1:],code[2:],code[3:]):
        if not (a.op==c.op==3 and a.args==c.args==(-4,4) and b.op in (0x1f,0x25) and d.op in (0x1f,0x25)):
            continue
        if incoming[b.pc]!={a.pc} or incoming[c.pc]!={b.pc} or incoming[d.pc]!={c.pc}:continue
        if b.target in (c.pc,d.pc) or d.target==d.end:continue
        forbidden=d.target if b.op==d.op else d.end
        surviving=d.end if b.op==d.op else d.target
        result[d.pc]={'pc':d.pc,'target':forbidden,'surviving_target':surviving,
            'first_test_pc':b.pc,'copies':[a.pc,c.pc],
            'reason':'two adjacent copies of the same unmodified condition; unique predecessor'}
    return result


def check_stack(code,funcs):
    """CFG height analysis with interprocedural stack summaries and action roots.

    Each function gets a net delta, a minimum required incoming SP depth, and
    checked equal-height joins. Native values and BP identities are not simulated.
    Recursive call cycles and paths with inconsistent stack effects fail closed.
    """
    by={i.pc:i for i in code};memo={};active=set();covered=set();callbacks=[];proven=proven_correlated_edges(code)
    for i in code:
        if i.op==5:call_signature(i,funcs)
        if i.op==0x2c:
            if any(v%4 for v in i.args):raise Refused(f'Unaligned stored frame at {i.pc}')
            callbacks.append(i)
    def aligned(*values):
        if any(v%4 for v in values):raise Refused('Unaligned stack access')
    def summary(entry):
        if entry in memo:return memo[entry]
        if entry in active:raise Refused(f'Recursive JSR summary not qualified at {entry}')
        active.add(entry);todo=deque([(entry,0)]);seen={};need=0;returns=set();bp_refs=[];maxheight=0
        while todo:
            pc,h=todo.popleft()
            if pc in seen:
                if seen[pc]!=h:raise Refused(f'Inconsistent CFG stack height at {pc}: {seen[pc]} vs {h}')
                continue
            if pc not in by:raise Refused(f'Fallthrough outside code at {pc}')
            seen[pc]=h;covered.add(pc);i=by[pc];op,q=i.op,i.qual
            delta=0;required=0;successors=None
            if op in (2,4):delta=4
            elif op in (1,3,0x26,0x27):
                off,size=i.args;aligned(off,size)
                if off>=0 or off+size>0:raise Refused(f'Copy extends above base at {pc}')
                if op in (1,3):required=max(-off,size if op==1 else 0)
                else:bp_refs.append((pc,off,size))
                if op in (3,0x27):delta=size
                else:required=max(required,size)
            elif op==5:
                f,p=call_signature(i,funcs);required=sum(SIZE[x['type']] for x in p)
                delta=SIZE[f['return']]-required
            elif 6<=op<=0x18:
                if q==0x24:required=2*i.args[0];aligned(i.args[0]);delta=4-required
                elif q==0x3a:required=24;delta=-12 if op>=0x14 else -20
                elif q in (0x3b,0x3c):required=16;delta=-4
                else:required=8;delta=-4
            elif op in (0x19,0x1a,0x22):required=4
            elif op==0x1b:
                delta=i.args[0];aligned(delta);required=max(0,-delta)
                if delta>0:raise Refused(f'Positive MOVSP not qualified at {pc}')
            elif op==0x1e:
                sub=summary(i.target);required=sub['need'];delta=sub['delta']
            elif op==0x1d:successors=[i.target]
            elif op in (0x1f,0x25):
                required=4;delta=-4;successors=[proven[pc]['surviving_target']] if pc in proven else [i.end,i.target]
            elif op==0x20:returns.add(h);successors=[]
            elif op==0x21:
                total,off,keep=i.args;aligned(total,off,keep)
                if off+keep>total:raise Refused(f'DESTRUCT retained slice outside frame at {pc}')
                required=total;delta=keep-total
            elif op in (0x23,0x24,0x28,0x29):
                off=i.args[0];aligned(off)
                if off>=0:raise Refused(f'Nonnegative INC/DEC offset at {pc}')
                if op in (0x23,0x24):required=-off
                else:bp_refs.append((pc,off,4))
            elif op==0x2a:delta=4
            elif op==0x2b:required=4;delta=-4
            elif op==0x2c:required=i.args[1]
            need=max(need,required-h);h+=delta;maxheight=max(maxheight,h)
            if successors is None:successors=[i.end]
            todo.extend((p,h) for p in successors)
        if len(returns)!=1:raise Refused(f'Subroutine {entry} has {len(returns)} distinct return stack effects')
        result={'entry':entry,'delta':next(iter(returns)),'need':need,'instructions':len(seen),'max_relative_depth':maxheight,'bp_accesses':len(bp_refs)}
        memo[entry]=result;active.remove(entry);return result
    root=summary(13)
    if root['need']>0:raise Refused(f'Entry stack underflow: needs {root["need"]} incoming bytes')
    if root['delta'] not in (0,4):raise Refused(f'Entry leaves {root["delta"]} bytes (expected void or int condition)')
    for i in callbacks:
        s=summary(i.target)
        # Captured locals provide the original relative-SP environment. The
        # callback is a void expression and must leave its capture stack intact.
        if s['need']>i.args[1]:raise Refused(f'Action capture underflow at {i.pc}: {s["need"]}>{i.args[1]}')
        if s['delta']!=0:raise Refused(f'Action callback at {i.target} changes captured stack by {s["delta"]}')
    # Check unreachable subroutine bodies too, never just the entry path.
    for i in code:
        if i.op==0x1e:summary(i.target)
    return {'status':'PASS','entry_delta':root['delta'],'subroutines':len(memo),
            'proven_infeasible_edges':list(proven.values()),'action_roots':len(callbacks),'instructions_checked':len(covered),
            'unreachable_instructions':len(code)-len(covered),
            'limits':['Height/bounds and equal-join analysis; not native value/type/handler execution or 2DA semantics.'],
            'summaries':sorted(memo.values(),key=lambda x:x['entry'])}


def _const(typ,value):
    if typ=='int':return bytes([4,3])+struct.pack('>i',int(value))
    if typ=='object':return bytes([4,6])+struct.pack('>I',int(value)&0xffffffff)
    if typ=='float':return bytes([4,4])+struct.pack('>f',float(value))
    if typ=='string':
        b=value.encode('cp1252')
        if len(b)>65535:raise Refused('Default string too long')
        return bytes([4,5])+struct.pack('>H',len(b))+b
    raise Refused('Unsupported appended default type: '+typ)


def _copy(op,off,size):return bytes([op,1])+struct.pack('>iH',off,size)
def _move(size):return b'\x1b\0'+struct.pack('>i',size)


def symbolic_frame_check(parts,old_params,new_params,ret):
    """Independent typed token execution of an inserted straight-line frame."""
    prefix=[('int','sentinel0'),('string','sentinel1')]
    old=[]
    for n,p in reversed(list(enumerate(old_params))):
        old.extend((p['type'],f'arg{n}:{c}') for c in range(SIZE[p['type']]//4))
    stack=prefix+old;seen_call=False
    blob=b''.join(parts);code=decode(b'NCS V1.0B'+struct.pack('>I',len(blob)+13)+blob+b'')
    for i in code:
        if i.op==4:stack.append(({3:'int',4:'float',5:'string',6:'object'}[i.qual],i.args[0]))
        elif i.op==3:
            off,n=i.args;start=len(stack)+off//4;stack.extend(stack[start:start+n//4])
        elif i.op==5:
            seen_call=True
            for n,p in enumerate(new_params):
                width=SIZE[p['type']]//4
                vals=stack[-width:] if width else []
                if any(t!=p['type'] for t,_ in vals):raise Refused('Adapter argument type mismatch')
                if n<len(old_params):
                    expected=[(p['type'],f'arg{n}:{c}') for c in range(width)]
                    if vals!=expected:raise Refused('Adapter changes original argument order/value')
                else:
                    expected_value=p['default']
                    if p['type']=='string':expected_value=expected_value.encode('cp1252')
                    if vals!=[(p['type'],expected_value)]:raise Refused('Adapter added default value mismatch')
                if width:del stack[-width:]
            stack.extend((ret,f'result:{c}') for c in range(SIZE[ret]//4))
        elif i.op==1:
            off,n=i.args;start=len(stack)+off//4;stack[start:start+n//4]=stack[-n//4:]
        elif i.op==0x1b:del stack[len(stack)+i.args[0]//4:]
        else:raise Refused('Unexpected opcode in adapter proof')
    expected=prefix+[(ret,f'result:{c}') for c in range(SIZE[ret]//4)]
    if not seen_call or stack!=expected:raise Refused('Adapter corrupts caller stack or result')
    return 'PASS: argument identities, typed defaults, stack prefix and returned cells preserved'


def convert_ncs(data,*,abi=None,source_game='k1'):
    abi=abi or load_abi();source=abi[source_game]['functions'];target=abi['k2']['functions']
    code=decode(data);before=check_stack(code,source)
    if source_game=='k2':
        return data,{'status':'unchanged','input_sha256':sha(data),'output_sha256':sha(data),'stack':before,'changes':[],'disassembly_diff':'identity','runtime':'UNTESTED'}
    dest={f['name']:f for f in target};pieces={};changes=[]
    for i in code:
        parts=[i.raw]
        if i.op==5:
            f,passed=call_signature(i,source);g=dest.get(f['name'])
            if g is None:raise Refused(f'No K2 routine for {f["name"]}; hand fix and native semantics required at {i.pc}')
            same_types=f['return']==g['return'] and [x['type'] for x in f['params']]==[x['type'] for x in g['params']]
            if same_types:
                parts=[b'\x05\0'+struct.pack('>HB',g['id'],i.args[1])]
                if g['id']!=f['id']:changes.append({'pc':i.pc,'function':f['name'],'kind':'routine-remap','from_id':f['id'],'to_id':g['id'],'argc':i.args[1]})
            else:
                if f['return']!=g['return'] or [p['type'] for p in g['params'][:len(f['params'])]]!=[p['type'] for p in f['params']]:
                    raise Refused('Non-prefix ABI change requires hand fix: '+f['name'])
                additions=[dict(p) for p in g['params'][len(passed):]]
                for n,p in enumerate(additions,start=len(passed)):
                    # An omitted old optional argument must keep its K1 default.
                    if n<len(f['params']):
                        p.update(default=f['params'][n].get('default'),default_expression=f['params'][n].get('default_expression'))
                    if p['default_expression'] is None:raise Refused('Required new argument has no default: '+f['name'])
                effective=g['params'][:len(passed)]+additions
                oldsize=sum(SIZE[p['type']] for p in passed)
                extra=sum(SIZE[p['type']] for p in additions)
                if any(p['type']=='action' for p in passed+additions):raise Refused('Action-valued changed ABI requires hand fix')
                parts=[_const(p['type'],p['default']) for p in reversed(additions)]
                if oldsize:parts.append(_copy(3,-oldsize-extra,oldsize))
                parts.append(b'\x05\0'+struct.pack('>HB',g['id'],len(g['params'])))
                if oldsize:
                    if SIZE[f['return']]:parts.append(_copy(1,-oldsize-SIZE[f['return']],SIZE[f['return']]))
                    parts.append(_move(-oldsize))
                proof=symbolic_frame_check(parts,passed,effective,f['return'])
                changes.append({'pc':i.pc,'function':f['name'],'kind':'typed-default-frame','from_id':f['id'],'to_id':g['id'],'from_argc':len(passed),'to_argc':len(effective),
                                'added_defaults':additions,'old_argument_bytes':oldsize,'return_bytes':SIZE[f['return']],'symbolic_proof':proof})
        pieces[i.pc]=parts
    mapping={};pc=13
    for i in code:mapping[i.pc]=pc;pc+=sum(map(len,pieces[i.pc]))
    output=[]
    for i in code:
        parts=pieces[i.pc]
        if i.op in JUMPS:parts=[i.raw[:2]+struct.pack('>i',mapping[i.target]-mapping[i.pc])]
        elif i.op==0x2c:
            offset=mapping[i.target]-mapping[i.pc]
            if not 0<offset<256:raise Refused('Relocated STORE_STATE target exceeds qualifier byte')
            parts=[bytes([i.op,offset])+i.raw[2:]]
        output.extend(parts)
    result=b'NCS V1.0B'+struct.pack('>I',pc)+b''.join(output)
    after_code=decode(result);after=check_stack(after_code,target)
    a={i.pc:i for i in after_code};diff=[]
    for i in code:
        j=a[mapping[i.pc]]
        if i.op in JUMPS|{0x2c}:
            if j.target!=mapping[i.target] or j.op!=i.op or (i.op==0x2c and j.raw[2:]!=i.raw[2:]):raise Refused('Control-flow disassembly mismatch')
        elif i.op!=5 and j.raw!=i.raw:raise Refused('Unexpected non-ACTION operand change')
        if pieces[i.pc]!=[i.raw]:diff.append({'source_pc':i.pc,'target_pc':mapping[i.pc],'source_hex':i.raw.hex(),'replacement_hex':b''.join(pieces[i.pc]).hex()})
    if before['entry_delta']!=after['entry_delta']:raise Refused('Converted entry stack balance differs')
    receipt={'status':'converted' if result!=data else 'unchanged','input_sha256':sha(data),'output_sha256':sha(result),
             'input_bytes':len(data),'output_bytes':len(result),'source_nwscript_sha256':abi['k1']['sha256'],'target_nwscript_sha256':abi['k2']['sha256'],
             'changes':changes,'stack_before':before,'stack_after':after,'disassembly_diff':diff,
             'jump_targets':'PASS: all source branch and stored-action edges map to corresponding instruction starts',
             'opcode_0x2d':False,'nop_padding_added':0,'runtime':'UNTESTED; offline ABI proof only'}
    return result,receipt


def scan_ncs(data,*,abi=None,game='k2'):
    abi=abi or load_abi();code=decode(data);return check_stack(code,abi[game]['functions'])


def convert_with_ghost(data, *, abi, resref='script'):
    """K1->K2 bytes plus receipt. Exact source route first; CFG proof required.

    abi must come from make_abi() over the player's read-only nwscript resources.
    The caller retains install/origin/2DA/native qualification authority.
    """
    from src.core.scripting.reference import inspect_ncs
    route={'attempted':True,'accepted':False,'k1_exact_recompile':False}
    recovered=None
    try:
        recovered=inspect_ncs(data,game='K1',resref=resref)
        route.update(k1_exact_recompile=recovered.exact_recompile,
                     recovery_error=recovered.recompile_error)
    except Exception as exc:
        route['recovery_error']=type(exc).__name__+': '+str(exc)[:1200]
    try:out,receipt=convert_ncs(data,abi=abi)
    except Refused as exc:
        exc.source_route=route
        raise
    receipt['route']='bytecode'
    if recovered is not None and recovered.exact_recompile:
        from pykotor.common.misc import Game
        from pykotor.resource.formats.ncs import compile_nss,bytes_ncs
        try:
            candidate=bytes(bytes_ncs(compile_nss(recovered.recovered_source,Game.K2)))
            decode(candidate)
            if candidate==out:
                route.update(accepted=True,proof='byte-identical to independently validated ABI rewrite')
                receipt['route']='ghost-source'
            else:route['cause']='K2 source compilation differs outside the checked ABI rewrite'
        except Exception as exc:route['cause']=type(exc).__name__+': '+str(exc)[:1200]
    receipt['source_route']=route
    return out,receipt
