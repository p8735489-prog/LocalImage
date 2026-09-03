#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GLSLC=${GLSLC:-glslc}
OUT="$ROOT/app/src/main/cpp/vulkan/generated"
mkdir -p "$OUT"
for src in "$ROOT"/app/src/main/shaders/nn/*.comp; do
  name=$(basename "$src" .comp)
  "$GLSLC" -O "$src" -o "$OUT/$name.spv"
  python3 - "$OUT/$name.spv" "$OUT/${name}_spirv.h" "$name" <<'PY'
import sys, struct
spv=open(sys.argv[1],'rb').read(); assert len(spv)%4==0
w=struct.unpack('<%dI'%(len(spv)//4),spv)
name=sys.argv[3]
with open(sys.argv[2],'w') as f:
 f.write('#pragma once\n#include <cstdint>\n#include <cstddef>\nnamespace localimage::vulkan::shader {\n')
 f.write('static constexpr uint32_t %s[] = {'%name)
 for i,x in enumerate(w):
  if i%8==0:f.write('\n ')
  f.write('0x%08xu,'%x)
 f.write('\n};\nstatic constexpr size_t %s_words = sizeof(%s)/sizeof(uint32_t);\n}\n'%(name,name))
PY
done
