#!/bin/csh -f
# Process the indirect-dimension of the reconstructed data
xyz2pipe -in ./A3DK08_recon.ft1 -x -verb \
| nmrPipe -fn TP \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 1 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 -di \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto \
| pipe2xyz -out ft2/test%05d.ft2 -x -ov

xyz2pipe -in ft2/test%05d.ft2 -z -verb \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 1 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 -di \
| pipe2xyz -out ft/test%03d.ft3 -z -ov

xyz2pipe -in ft/test%03d.ft3 -z -verb \
  > ./A3DK08_recon.ft

/bin/rm -rf ft2



