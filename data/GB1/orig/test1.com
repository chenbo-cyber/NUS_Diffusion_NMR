#!/bin/csh

nmrDraw -in test.ft2 -broadcast -group 1 `getGeom -west` -nophase -verb 2 -dbg 2 


