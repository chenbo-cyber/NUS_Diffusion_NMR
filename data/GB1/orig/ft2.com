#!/bin/csh

set procArgs = "-xP0 -48 -xEXTX1 10.8ppm -xEXTXN 5.8ppm -yFTARG alt -scaleTo 100.0"

basicFT2.com -in test.fid  -out test.ft2 $procArgs -title US_FT -yZFARG zf=2,auto

