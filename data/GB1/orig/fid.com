#!/bin/csh

bruk2pipe -in ./ser \
  -bad 0.0 -ext -noaswap -AMX -decim 16 -dspfvs 12 -grpdly -1  \
  -xN              3584  -yN               340  \
  -xT              1676  -yT               170  \
  -xMODE            DQD  -yMODE        Complex  \
  -xSW         8389.262  -ySW         2128.565  \
  -xOBS         600.133  -yOBS          60.818  \
  -xCAR           4.773  -yCAR         118.562  \
  -xLAB              HN  -yLAB             15N  \
  -ndim               2  -aq2D         Complex  \
  -out ./test.fid -verb -ov

