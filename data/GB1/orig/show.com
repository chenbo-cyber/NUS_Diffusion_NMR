#!/bin/csh

set fList = (test.ft2 us50.ft2 \
             us_lp.ft2 us50_lp.ft2 \
             us_nuszf_ist.ft2 us_nuszf_rist.ft2 us_nuszf_smile.ft2 \
             us50_nuszf_ist.ft2 us50_nuszf_rist.ft2 us50_nuszf_smile.ft2 \
             nus50_ist.ft2 nus50_rist.ft2 nus50_nuszf_ist.ft2 \
             nus50_nuszf_rist.ft2 nus50_nuszf_smile.ft2 nus50.ft2)

specView.tcl $* -hi 5.0 -ref None -in $fList 

