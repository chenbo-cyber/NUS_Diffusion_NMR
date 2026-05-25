#!/bin/csh

fid.com
ft2.com

autoFit.tcl -inSpec test.ft2 -inTab master.tab -noexec
autoFit.com

nmrDraw -broadcast -nophase -group 1 `getGeom -left`  -in test.ft2 -hi 5.0 &
sleep 3

nmrDraw -broadcast -nophase -group 1 `getGeom -mid`   -in sim.ft2  -hi 5.0 &
sleep 3

nmrDraw -broadcast -nophase -group 1 `getGeom -right` -in dif.ft2  -hi 5.0 &
