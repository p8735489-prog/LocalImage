#!/usr/bin/env python3
import struct, sys
if len(sys.argv) != 4: raise SystemExit("usage: spv_to_header.py input.spv output.h symbol")
raw=open(sys.argv[1],'rb').read()
if len(raw)%4: raise SystemExit("SPIR-V size is not word aligned")
words=struct.unpack('<%dI'%(len(raw)//4),raw)
symbol=sys.argv[3]
with open(sys.argv[2],'w',encoding='utf-8') as f:
    f.write('#pragma once\n#include <cstdint>\n#include <cstddef>\nnamespace localimage::vulkan::shader {\n')
    f.write('static constexpr uint32_t %s[] = {'%symbol)
    for i,w in enumerate(words):
        if i%8==0: f.write('\n    ')
        f.write('0x%08Xu,'%w)
    f.write('\n};\n')
    f.write('static constexpr size_t %s_words = sizeof(%s)/sizeof(uint32_t);\n'%(symbol,symbol))
    f.write('}\n')
