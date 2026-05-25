#!/bin/csh

set inName   = (`getArgD $argv -in ist.ft2`)
set maskName = (`getArgD $argv -in mask.fid`)
set outName  = (`getArgD $argv -in ist_exact.ft2`)

nmrPipe -in $inName \
| nmrPipe -fn TP \
| nmrPipe -fn HT -auto \
| nmrPipe -fn FT -inv \
| nmrPipe -fn ZF -inv \
| nmrPipe -fn APOD -hdr -inv \
  -out ist.ft1 -ov
