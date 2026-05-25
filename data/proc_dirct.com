#!/bin/csh

# 
# Apply windowing to the direct-dimension data of A3DK08
# perform FT, apply phase correction, and discard the imaginary component
# 

xyz2pipe -in data/test%03d.fid -x -verb \
| nmrPipe -fn SOL \
| nmrPipe -fn SP -off 0.5 -end 0.95 -pow 2 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 -140.0 -p1 -1.0 -di \
| nmrPipe -fn EXT -x1 3% -xn 47% \
| pipe2xyz -out ft2/test%05d.ft -y -ov

xyz2pipe -in ft2/test%05d.ft -y -verb \
  > ./A3DK08_label.ft1

/bin/rm -rf ft2


