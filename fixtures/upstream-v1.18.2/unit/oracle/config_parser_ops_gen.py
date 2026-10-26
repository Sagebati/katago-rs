#!/usr/bin/env python3
"""Generates config_parser_ops.txt, the script run by BOTH oracle/config_parser.cpp (against upstream's
ConfigParser) and crates/katago-core/tests/config_parser.rs (against the port). See the header of
config_parser.cpp for the op set. Run from anywhere:
  python3 -I fixtures/upstream-v1.18.2/unit/oracle/config_parser_ops_gen.py > fixtures/upstream-v1.18.2/unit/oracle/config_parser_ops.txt
"""
import struct
import sys

out = []


def esc(s):
    if isinstance(s, str):
        s = s.encode('utf-8', 'surrogateescape')
    r = []
    for c in s:
        ch = chr(c)
        if ch == '\\':
            r.append('\\\\')
        elif ch == '\n':
            r.append('\\n')
        elif ch == '\r':
            r.append('\\r')
        elif ch == '\t':
            r.append('\\t')
        elif c < 0x20 or c >= 0x7f:
            r.append('\\x%02x' % c)
        else:
            r.append(ch)
    return ''.join(r)


def op(*fields):
    out.append('\t'.join([fields[0]] + [esc(f) for f in fields[1:]]))


def comment(s):
    out.append('# ' + s)


def fbits(x):
    return '%08X' % struct.unpack('>I', struct.pack('>f', x))[0]


def dbits(x):
    return '%016X' % struct.unpack('>Q', struct.pack('>d', x))[0]


# ---------------------------------------------------------------------------------------------
comment('S1: single-line matrix: pre + key + sep + value + post, stream constructor, default flags')
pres = ['']
keys = ['', 'k', 'K_1-x', '0a', 'a b', 'a$', 'a#', '#k', 'é', 'a.b', '-', '_']
seps = ['', '=', ' = ', '=  ', '  =', ' \t= \t', '==', ': ', ' ']
vals = ['', 'v', 'v w', '"q"', '"q w"', '""', '"a\\"b"', '"a\\\\b"', '"a\\', '"abc', 'x"y"', '# c', 'v # c', 'v#c',
        '"q" # c', '"q"#c', '"q" x', '"q""r"', 'a=b', 'é', '"é\\é"', "'sq'", 'v \\', '\\#', '"\\', '"q"  ', ' "q"', '"#"', '" # "',
        '"a#b" #c', '1 2 3', '\x0b', 'v\x0c']
n = 0
for k in keys:
    for s in seps:
        for v in vals:
            op('new', 'm', '0', '1', k + s + v + '\n')
            n += 1
comment('S1b: surrounding whitespace, line endings and trailing text around a few key/value shapes')
pres = ['', ' ', '\t', '  \t', '\x0b', '\x0c']
posts = ['', ' ', '\r', '\r\n', '\n\n', ' # tail\n', '\n#c', '\n  \n', '\nz=1', '\nz=1\n', '\n=', '\n"x"']
cores = ['k=v', 'k = "v w" # c', 'k =v#c', '"k"=v', 'k=', '#k=v', 'k', 'k v', '@include']
for p in pres:
    for c in cores:
        for q in posts:
            op('new', 'm', '0', '1', p + c + q)

# ---------------------------------------------------------------------------------------------
comment('S2: whole-config shapes under all four flag combinations')
texts = [
    '', '\n', '\n\n\n', '# only comment', '# only comment\n', 'a=1', 'a=1\n', 'a=1\r\nb=2\r\n', 'a=1\rb=2', 'a = 1\n\n\n  b=2  \n',
    'a=1\na=2\n', 'a=1\n#x\nb=2\na=3\n', 'a=1\nb=2\na=3\nb=4\n',
    'b=1\na=2\nB=3\n_c=4\n-d=5\n0e=6\na-b=7\na_b=8\naB=9\nz=10\nZ=11\n',
    'a=1\x00b=2\n', 'a=1\n\x00\nb=2\n', '\x0b\x0c a=1 \x0b\x0c\n', 'a=é\n', 'é=1\n', '﻿a=1\n',
    'a=' + 'x' * 5000 + '\n', 'k' * 3000 + '=v\n', 'a=\t1\t\n', '\ta\t=\t1\t#c\n', '  # comment with = and " quotes\n',
    'a=1\n' * 50, ''.join('k%d = %d\n' % (i, i * i) for i in range(300)),
    'a=1\n  \n\t\n#\n##\n b = 2\n', '\n\n  a  =  "  spaced  "  \n\n', 'a="x"\nb=" y "\nc=" \\" "\n',
    'trailing=newline is okay', 'a=1\nb', 'a=1\nb=\n', 'a=1\n=b\n', 'a=1\n"b"=2\n', 'a=1 b=2\n', 'a=1,b=2\n',
]
for t in texts:
    for ko in '01':
        for kfi in '01':
            op('new', 's2', ko, kfi, t)

# ---------------------------------------------------------------------------------------------
comment('S3: @include (files written to the scratch directory by the oracle, served from memory by the Rust test)')
op('file', 's1/a.cfg', 'x=1\n@include b.cfg\ny=2\n')
op('file', 's1/b.cfg', 'z=3\n')
for ko in '01':
    for kfi in '01':
        op('newfile', 'f', ko, kfi, 's1/a.cfg')
op('file', 's2/a.cfg', 'x=1\n@include b.cfg\n')
op('file', 's2/b.cfg', 'x=2\n')
for ko in '01':
    for kfi in '01':
        op('newfile', 'f', ko, kfi, 's2/a.cfg')
op('file', 's3/a.cfg', '@include b.cfg\nx=9\n')
op('file', 's3/b.cfg', 'x=2\ny=3\n')
for ko in '01':
    for kfi in '01':
        op('newfile', 'f', ko, kfi, 's3/a.cfg')
op('file', 's4/a.cfg', '@include d/e.cfg\na=1\n')
op('file', 's4/d/e.cfg', '@include f.cfg\ne=1\n')
op('file', 's4/d/f.cfg', 'f=1\n')
op('file', 's4/d/g.cfg', '@include ../h.cfg\ng=1\n@include f.cfg\n')
op('file', 's4/h.cfg', 'h=1\n')
op('newfile', 'f', '0', '1', 's4/a.cfg')
op('newfile', 'f', '0', '1', 's4/d/g.cfg')
op('newfile', 'f', '0', '1', 's4/d/e.cfg')
op('file', 's5/c0.cfg', '@include c1.cfg\nc0=1\n')
op('file', 's5/c1.cfg', 'c1=1\n@include c0.cfg\n')
op('newfile', 'f', '0', '1', 's5/c0.cfg')
op('file', 's5/self.cfg', 'a=1\n@include self.cfg\n')
op('newfile', 'f', '0', '1', 's5/self.cfg')
op('file', 'self2.cfg', 'a=1\n@include self2.cfg\n')
op('newfile', 'f', '0', '1', 'self2.cfg')
op('file', 's5/m.cfg', '@include z.cfg\n@include z.cfg\n')
op('file', 's5/z.cfg', 'z=1\n')
op('newfile', 'f', '0', '1', 's5/m.cfg')
op('file', 's5/m2.cfg', '@include z.cfg\n@include ./z.cfg\n')
op('newfile', 'f', '0', '1', 's5/m2.cfg')
op('file', 's6/abs.cfg', '@include /abs/x.cfg\n')
op('newfile', 'f', '0', '1', 's6/abs.cfg')
op('file', 's6/abs2.cfg', 'a=1\n@include \\x\\y.cfg\n')
op('newfile', 'f', '0', '1', 's6/abs2.cfg')
op('file', 's6/missing.cfg', 'a=1\n@include nope.cfg\n')
op('newfile', 'f', '0', '1', 's6/missing.cfg')
op('newfile', 'f', '0', '1', 'nofile.cfg')
op('newfile', 'f', '0', '1', 'nodir/nofile.cfg')
op('file', 's7/a.cfg', '@include b.cfg\nbad line here\n')
op('file', 's7/b.cfg', 'b=1\n')
op('newfile', 'f', '0', '1', 's7/a.cfg')
op('file', 's7/c.cfg', 'c=1\n\nc=2\n')
op('file', 's7/d.cfg', 'a=1\n@include c.cfg\nd=1\n')
for ko in '01':
    op('newfile', 'f', ko, '1', 's7/d.cfg')
op('file', 's7/e.cfg', '\n\n@include c.cfg\n\nbad\n')
op('newfile', 'f', '0', '1', 's7/e.cfg')
op('file', 's7/crlf.cfg', 'a=1\r\n@include b.cfg\r\nc=3\r\n')
op('newfile', 'f', '0', '1', 's7/crlf.cfg')
op('file', 's7/nonl.cfg', 'a=1\n@include b.cfg')
op('newfile', 'f', '0', '1', 's7/nonl.cfg')
op('file', 's7/empty.cfg', '')
op('newfile', 'f', '0', '1', 's7/empty.cfg')
op('file', 's7/q.cfg', '@include "b.cfg"\n@include \'z.cfg\'\n')
op('file', 's7/z.cfg', 'z=1\n')
op('newfile', 'f', '0', '1', 's7/q.cfg')
op('file', 's7/ie.cfg', 'a=1  # comment\n@include b.cfg # trailing\n')
op('newfile', 'f', '0', '1', 's7/ie.cfg')

comment('S3b: @ directive syntax in a stream (includes resolved relative to the scratch directory)')
directives = [
    '@include s1/b.cfg', '@include "s1/b.cfg"', "@include 's1/b.cfg'", '@include = s1/b.cfg', '@include\ts1/b.cfg # c',
    '@include s1/b.cfg#c', '@include=s1/b.cfg', '@includes1/b.cfg', '@include', '@include ', '@include =', '@include   =  ',
    '@foo s1/b.cfg', '@ include s1/b.cfg', '@include "s1/b.cfg', '@include s1/b.cfg"', '  @include s1/b.cfg  ',
    '@include\x0bs1/b.cfg', '@\tinclude s1/b.cfg', '#@include s1/b.cfg', 'a=@include s1/b.cfg', '@include s1/b.cfg extra',
    '@include s1/b.cfg\n@include s1/b.cfg', '@include s1/b.cfg\nz=4', 'z=4\n@include s1/b.cfg', '@include ./s1/b.cfg',
    '@include s1/../s1/b.cfg', '@include s1//b.cfg', '@include /s1/b.cfg', '@include \\s1\\b.cfg', '@include s1\\b.cfg',
    '@include \'\'', '@include ""', '@include "\'s1/b.cfg\'"', '@include \'"s1/b.cfg"\'', '@abcdefgh',
    '@includ=', '@include#', '@include s1/a.cfg', '@include nope.cfg', '@include s1/b.cfg\n@include s1/a.cfg',
]
for d in directives:
    for ko in '01':
        op('new', 'd', ko, '1', d + '\n')
op('new', 'd', '0', '0', '@include s1/b.cfg\nz=4\n')
op('new', 'd', '0', '0', 'z=4\n@include s1/b.cfg\n')

comment('S3c: overrideKeys(file) / initialize(file) / copy')
op('newfile', 'o', '0', '1', 's1/a.cfg')
op('overridekeysfile', 'o', 's3/b.cfg')
op('overridekeysfile', 'o', 's3/b.cfg')
op('overridekeysfile', 'o', 's2/b.cfg')
op('overridekeysfile', 'o', 'nope.cfg')
op('overridekeysfile', 'o', 's1/b.cfg')
op('overridekeysfile', 'o', 's4/d/f.cfg')
op('overridekeysfile', 'o', '/abs/x.cfg')
op('copy', 'oc', 'o')
op('info', 'oc')
op('new', 'p', '0', '1', 'a=1\n')
op('overridekeysfile', 'p', 's1/b.cfg')
op('overridekeysfile', 'p', 's1/b.cfg')
op('overridekeysfile', 'p', 's4/d/f.cfg')
op('overridekeysfile', 'p', 's4/a.cfg')
op('newempty', 'e', '0', '1')
op('init_file', 'e', 's1/a.cfg')
op('init_file', 'e', 's1/a.cfg')
op('init_in', 'e', 'a=1')
op('init_kvs', 'e', 'a=1')
op('newempty', 'e2', '1', '0')
op('init_file', 'e2', 'nofile.cfg')
op('init_file', 'e2', 's7/c.cfg')
op('init_in', 'e2', 'a=1')
op('newempty', 'e3', '0', '1')
op('init_in', 'e3', 'x=1\n@include s1/b.cfg\n')
op('overridekeysfile', 'e3', 's1/b.cfg')
op('overridekeysfile', 'e3', 's2/b.cfg')

# ---------------------------------------------------------------------------------------------
comment('S4: getters')
G = '''# getters
s = hello world
sq = "quoted # string"
sempty_list = a,,b, ,c
sl = a , b,c
bt = true
bf = FALSE
bsp =  Tr
bsp2 =   True
byes = yes
b1 = 1
bl = true, false,TRUE, False
blbad = true,maybe
e1 = auto
e2 = Enabled
e3 = no
e4 = maybe
e5 =  AUTO
e6 = T
e7 = 1
e8 = disabled
e9 = Y
i = 42
ineg = -17
iplus = +5
izero = 0
imax = 2147483647
imin = -2147483648
iover = 2147483648
iunder = -2147483649
ihuge = 99999999999
ihex = 0x10
ispace =  7
ifloaty = 3.5
iempty_str = abc
i2 = 1,2, 3 ,4
i2bad = 1,x
i2range = 5,50,500
i2sp = 1, 2 ,3
i64 = -9000000000000
i64max = 9223372036854775807
i64min = -9223372036854775808
i64over = 9223372036854775808
i64huge = 99999999999999999999
i64l = 1, -2 , 9000000000000
u64 = 18446744073709551615
u64mid = 18000000000000
uneg = -1
uzero = 0
uover = 18446744073709551616
u64l = 0, 1, 18446744073709551615
u64lbad = 1,-1
f = 1.5
fneg = -0.0
fnan = nan
fnan2 = NaN
finf = inf
fninf = -inf
fhuge = 1e39
fexp = 1.5e20
fsmall = 1e-50
fsub = 1e-40
fprec = 0.1
fhexf = 0x1p3
fbad = 1.5x
fl = -1.5, 0.0, 1.5
flnan = 1.0,nan
fldbl = 1e39
d = 2.718281828459045
dnan = NaN
dinf = -inf
dhuge = 1e400
dsub = 4.9e-324
dbig = 1.7976931348623157e308
dprec = 0.1
dbad = abc
dl = -1.0e50, 0.0, 1.0e50
dlbad = 1,2,x
pairs = 1-2, 3-4,,5-6
pairs1 = 7-8
pairs_bad3 = 1-2-3
pairs_badA = a-b
pairs_bad1 = 1-
pairs_badneg = -5-6
pairs_range = 1-2, 3-400
pairs_blank = ,, ,
pairs_sp = 1 - 2, 3 -4
dup_a = 1
Zkey = Z
'''
op('new', 'g', '0', '1', G)
op('info', 'g')

ints_ranges = [('lowest', 'max'), ('0', '100'), ('50', '100'), ('-100', '0'), ('42', '42'), ('43', '100'), ('0', '41'), ('-20', '-10')]
int_keys = ['i', 'ineg', 'iplus', 'izero', 'imax', 'imin', 'iover', 'iunder', 'ihuge', 'ihex', 'ispace', 'ifloaty', 'iempty_str',
            'i64', 'u64', 's', 'nokey', 'f', 'bt']
for k in int_keys:
    for lo, hi in ints_ranges:
        op('getint', 'g', k, lo, hi)
for k in ['i2', 'i2bad', 'i2range', 'i2sp', 'i', 'nokey', 's', 'ineg']:
    for lo, hi in ints_ranges:
        op('getints', 'g', k, lo, hi)

i64_ranges = [('lowest', 'max'), ('0', '100'), ('-9000000000000', '0'), ('-9000000000000', '-9000000000000'), ('0', '9223372036854775807'),
              ('-9223372036854775808', '-1')]
for k in ['i', 'ineg', 'i64', 'i64max', 'i64min', 'i64over', 'i64huge', 'u64', 'ihex', 's', 'nokey', 'ispace']:
    for lo, hi in i64_ranges:
        op('getint64', 'g', k, lo, hi)
for k in ['i64l', 'i2', 'i2bad', 'nokey']:
    for lo, hi in i64_ranges:
        op('getint64s', 'g', k, lo, hi)

u64_ranges = [('lowest', 'max'), ('0', '100'), ('1', '18000000000000'), ('18000000000000', '18446744073709551615'), ('18446744073709551615', '18446744073709551615'),
              ('1', '1')]
for k in ['u64', 'u64mid', 'uneg', 'uzero', 'uover', 'i', 'ineg', 'ihex', 's', 'nokey', 'ispace']:
    for lo, hi in u64_ranges:
        op('getuint64', 'g', k, lo, hi)
for k in ['u64l', 'u64lbad', 'i2', 'nokey']:
    for lo, hi in u64_ranges:
        op('getuint64s', 'g', k, lo, hi)

f_ranges = [('lowest', 'max'), (fbits(0.0), fbits(100.0)), (fbits(1.0), fbits(2.0)), (fbits(-2.0), fbits(-1.0)), (fbits(1.5), fbits(1.5)),
            (fbits(0.5), fbits(1.25)), (fbits(-1e30), fbits(1e30))]
for k in ['f', 'fneg', 'fnan', 'fnan2', 'finf', 'fninf', 'fhuge', 'fexp', 'fsmall', 'fsub', 'fprec', 'fhexf', 'fbad', 'i', 'd', 'dhuge', 's', 'nokey']:
    for lo, hi in f_ranges:
        op('getfloat', 'g', k, lo, hi)
for k in ['fl', 'flnan', 'fldbl', 'i2', 'f', 'nokey']:
    for lo, hi in f_ranges:
        op('getfloats', 'g', k, lo, hi)

d_ranges = [('lowest', 'max'), (dbits(0.0), dbits(100.0)), (dbits(1.0), dbits(2.0)), (dbits(-2.0), dbits(-1.0)), (dbits(2.718281828459045), dbits(2.718281828459045)),
            (dbits(0.5), dbits(1.25)), (dbits(-1e300), dbits(1e300))]
for k in ['d', 'f', 'fneg', 'dnan', 'dinf', 'dhuge', 'dsub', 'dbig', 'dprec', 'dbad', 'fnan', 'finf', 'i', 's', 'nokey', 'fhuge']:
    for lo, hi in d_ranges:
        op('getdouble', 'g', k, lo, hi)
for k in ['dl', 'dlbad', 'fl', 'i2', 'd', 'nokey']:
    for lo, hi in d_ranges:
        op('getdoubles', 'g', k, lo, hi)

for k in ['s', 'sq', 'sempty_list', 'sl', 'bt', 'i', 'nokey', 'Zkey', 'pairs']:
    op('getstring', 'g', k)
    op('getstrings', 'g', k)
    op('getstringsnet', 'g', k)
for k in ['s', 'bt', 'e1', 'nokey', 'sl']:
    for poss in ['hello world', 'a,b,c', 'true,false', '', 'auto,yes', 'hello world,x']:
        op('getstringp', 'g', k, poss)
        op('getstringsp', 'g', k, poss)
for k in ['bt', 'bf', 'bsp', 'bsp2', 'byes', 'b1', 'bl', 'blbad', 'e1', 's', 'i', 'nokey']:
    op('getbool', 'g', k)
    op('getbools', 'g', k)
for k in ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'bt', 'bf', 'bsp', 'bsp2', 'b1', 's', 'nokey']:
    op('getenabled', 'g', k)
pair_ranges = [('lowest', 'max'), ('0', '100'), ('0', '5'), ('2', '8'), ('1', '400'), ('0', '0')]
for k in ['pairs', 'pairs1', 'pairs_bad3', 'pairs_badA', 'pairs_bad1', 'pairs_badneg', 'pairs_range', 'pairs_blank', 'pairs_sp', 'i', 'nokey']:
    for lo, hi in pair_ranges:
        op('getpairs', 'g', k, lo, hi)
op('info', 'g')

comment('S4b: the same getters under a file name (error texts embed it) and after overrideKeys changes it')
op('file', 'g/getters.cfg', 'i = 42\nf = 1.5\nbt = true\nbad = x\n')
op('newfile', 'gf', '0', '1', 'g/getters.cfg')
op('getint', 'gf', 'bad', 'lowest', 'max')
op('getint', 'gf', 'nokey', 'lowest', 'max')
op('getint', 'gf', 'i', '0', '10')
op('getfloat', 'gf', 'i', fbits(0.0), fbits(10.0))
op('getdouble', 'gf', 'f', dbits(2.0), dbits(3.0))
op('getbool', 'gf', 'bad')
op('getenabled', 'gf', 'bad')
op('getstringp', 'gf', 'bt', 'x,y')
op('overridekeys', 'gf', 'i=7')
op('getint', 'gf', 'i', '0', '5')
op('getint', 'gf', 'nokey', 'lowest', 'max')
op('overridekeys', 'gf', 'i=8')
op('getint', 'gf', 'nokey', 'lowest', 'max')

# ---------------------------------------------------------------------------------------------
comment('S5: contains, firstFound, overrides, alias, used keys, warnUnusedKeys, copy, kvs, parseCommaSeparated')
op('new', 'c', '0', '1', 'a=1\nb=2\nc-d=3\nprefix_x=4\nprefix_y=5\nprefixz=6\nother=7\n')
op('contains', 'c', 'a')
op('contains', 'c', 'A')
op('contains', 'c', '')
op('containsany', 'c', '')
op('containsany', 'c', 'x,y,z')
op('containsany', 'c', 'x,b')
op('containsany', 'c', 'c-d')
op('firstfail', 'c', 'x,y')
op('firstfail', 'c', 'x,b,a')
op('firstfail', 'c', '')
op('firstempty', 'c', 'x,y')
op('firstempty', 'c', 'x,b,a')
op('firstempty', 'c', '')
op('info', 'c')
op('getint', 'c', 'a', 'lowest', 'max')
op('info', 'c')
op('warn', 'c')
op('markused', 'c', 'b')
op('markused', 'c', 'not_a_key')
op('markprefix', 'c', 'prefix_')
op('info', 'c')
op('warn', 'c')
op('markprefix', 'c', '')
op('warn', 'c')
op('unsetused', 'c', 'a')
op('unsetused', 'c', 'nokey')
op('warn', 'c')
op('markprefix', 'c', 'zzz')
op('markprefix', 'c', 'o')
op('info', 'c')

op('newfile', 'cf', '0', '1', 's1/a.cfg')
op('warn', 'cf')
op('overridekeys', 'cf', 'q=1')
op('warn', 'cf')

op('copy', 'cc', 'c')
op('info', 'cc')
op('getstring', 'cc', 'a')
op('info', 'c')
op('newempty', 'ne', '0', '1')
op('copy', 'cne', 'ne')
op('info', 'ne')
op('getstring', 'ne', 'a')
op('contains', 'ne', 'a')
op('warn', 'ne')
op('newkvs', 'kv', 'a=1,b=2,c=x y')
op('info', 'kv')
op('getint', 'kv', 'a', 'lowest', 'max')
op('getstring', 'kv', 'c')
op('init_in', 'kv', 'a=1')
op('init_kvs', 'kv', 'a=2')
op('newkvs', 'kv0', '')
op('info', 'kv0')
op('copy', 'kvc', 'kv')
op('info', 'kvc')

op('new', 'ov', '0', '1', 'a=1\nb=2\nc=3\n')
op('overridekey', 'ov', 'a', '10')
op('overridekey', 'ov', 'b', '')
op('overridekey', 'ov', 'nokey', '')
op('overridekey', 'ov', 'd', '4')
op('overridekey', 'ov', 'e', ' ')
op('overridekey', 'ov', '', 'v')
op('overridekeys', 'ov', 'a=11,c=,x=1')
op('overridekeys', 'ov', '')
op('overridekeys', 'ov', 'a=,x=')
op('overridekeys', 'ov', 'q=r,s=t')
op('new', 'om', '0', '1', 'a1=1\na2=2\nb1=3\nb2=4\nc=5\nd=6\n')
op('overridekeysm', 'om', 'a1=9', 'a1,a2|b1,b2')
op('copy', 'om2', 'om')
op('overridekeysm', 'om2', 'b2=9', 'a1,a2|b1,b2')
op('copy', 'om3', 'om')
op('overridekeysm', 'om3', 'c=9', 'a1,a2|b1,b2')
op('copy', 'om4', 'om')
op('overridekeysm', 'om4', 'a2=1,b1=2', 'a1,a2|b1,b2')
op('copy', 'om5', 'om')
op('overridekeysm', 'om5', 'c=1', 'c|d;a1|b1')
op('overridekeysm', 'om5', 'd=', 'c|d;a1|b1')
op('overridekeysm', 'om5', 'a1=2,x=y', 'c|d;a1|b1;x|a2')
op('overridekeysm', 'om5', 'zz=1', '')
op('overridekeysm', 'om5', 'zz=2', 'zz|')
op('overridekeysm', 'om5', 'q=2', '|zz')

op('new', 'al', '0', '1', 'old=1\nnew=2\nonly=3\nused=4\n')
op('alias', 'al', 'old', 'new')
op('alias', 'al', 'old', 'renamed')
op('alias', 'al', 'nokey', 'other')
op('getint', 'al', 'used', 'lowest', 'max')
op('alias', 'al', 'used', 'usedtoo')
op('info', 'al')
op('alias', 'al', 'only', 'only')
op('alias', 'al', 'renamed', 'used')
op('info', 'al')

for s in ['', 'a=1', 'a=1,b=2', ' a = 1 , b = 2 ', 'a=1,,b=2', ',', ' , ', 'a', 'a=1,b', 'a=', '=b', ' = ', 'a=b=c', 'a=1,a=2', 'a = x y , b = "q"',
          'k=v,', ',k=v', 'a=1,\t,b=2', 'é=ü', 'a=1;b=2', 'a=1\nb=2', 'x=1,y=2,z', '=,=', 'b=2,a=1,A=3']:
    op('parsecomma', s)

sys.stdout.write('\n'.join(out) + '\n')
