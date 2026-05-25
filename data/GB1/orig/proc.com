#!/bin/csh

#
# Demonstration of different NMRPipe processing schemes, including Linear
# Prediction, and reconstruction methods SMILE and IST. Details are given
# at the end of the script.

#
# Processing arguments common to all schemes; 
#  Results are scaled to max 100.0 for easier comparison.

set procArgs = "-xP0 -48 -xEXTX1 10.8ppm -xEXTXN 5.8ppm -yFTARG alt -scaleTo 100.0"

#
# Generate a 50% truncated and 50% NUS version of the original US data:

nmrPipe -in test.fid \
| nmrPipe -fn TP \
| nmrPipe -fn EXT -xn 50% -sw \
| nmrPipe -fn TP \
  -out trunc.fid -ov
 
fakeNUS.com -in test.fid -nus 0.5 > fake.nuslist
nusCompress.tcl -mode pipe -in test.fid -out nus.fid -mask mask.fid -nocompress -sample fake.nuslist

#
#  Zero fill (-yZFARG) is adjusted in each scheme so that all spectra are the same size:

basicFT2.com -in test.fid  -out test.ft2              $procArgs -title US_FT             -yZFARG zf=2,auto
basicFT2.com -in trunc.fid -out us50.ft2              $procArgs -title US50_FT           -yZFARG zf=3,auto
basicFT2.com -in nus.fid   -out nus50.ft2             $procArgs -title NUS50_FT          -yZFARG zf=2,auto

basicFT2.com -in test.fid  -out us_lp.ft2             $procArgs -title US_LP             -yLP fb,ord=12
basicFT2.com -in trunc.fid -out us50_lp.ft2           $procArgs -title US50_LP           -yLP fb,ord=12 -yZFARG zf=2,auto

basicFT2.com -in test.fid  -out us_nuszf_ist.ft2      $procArgs -title US_IST_NUSZF      -nusZF -ist
basicFT2.com -in trunc.fid -out us50_nuszf_ist.ft2    $procArgs -title US50_IST_NUSZF    -nusZF -ist -yZFARG zf=2,auto

basicFT2.com -in test.fid  -out us_nuszf_rist.ft2     $procArgs -title US_RIST_NUSZF     -nusZF -ist -retain
basicFT2.com -in trunc.fid -out us50_nuszf_rist.ft2   $procArgs -title US50_RIST_NUSZF   -nusZF -ist -yZFARG zf=2,auto -retain

basicFT2.com -in test.fid  -out us_nuszf_smile.ft2    $procArgs -title US_SMILE_NUSZF    -nusZF -smile
basicFT2.com -in trunc.fid -out us50_nuszf_smile.ft2  $procArgs -title US50_SMILE_NUSZF  -nusZF -smile -yZFARG zf=2,auto

ist2D.com    -in nus.fid   -out nus50_ist.ft2         $procArgs -title NUS50_IST         -mask mask.fid -yZFARG zf=2,auto
ist2D.com    -in nus.fid   -out nus50_nuszf_ist.ft2   $procArgs -title NUS50_IST_NUSZF   -mask mask.fid -nusZF

ist2D.com    -in nus.fid   -out nus50_rist.ft2        $procArgs -title NUS50_RIST        -mask mask.fid -yZFARG zf=2,auto -retain
ist2D.com    -in nus.fid   -out nus50_nuszf_rist.ft2  $procArgs -title NUS50_RIST_NUSZF  -mask mask.fid -nusZF -retain

smile2D.com  -in nus.fid   -out nus50_nuszf_smile.ft2 $procArgs -title NUS50_SMILE_NUSZF -mask mask.fid -sample fake.nuslist -nusZF

#
# Demonstration of different processing schemes. Key:
#
#  test.fid               Conventional Uniformly-Sampled (US) time-domain data (see fid.com).
#  trunc.fid              Time-domain data truncated to 50% in the indirect dimension
#  nus.fid                50% Non-Uniformly Sampled (NUS) data made by resampling test.fid.
#
#  fake.nuslist           NUS Sampling Schedule text file (here, created by fakeNUS.com).
#  mask.fid               NMRPipe-format NUS Mask File (here, created by nusCompress.tcl).
#
#  test.ft2               FT (Fourier Transform) of original US data.
#  trunc.ft2              FT of truncated US data.
#  nus.ft2                FT of NUS Data.
#
#  us_lp.ft2              FT of original US data with Linear Prediction (time-domain extrapolation).
#  us50_lp.ft2            FT of truncated US data with Linear Prediction.
#
#  us_nuszf_ist.ft2       IST NUS Zero Fill (time-domain extrapolation) of original US data.
#  us50_nuszf_ist.ft2     IST NUS Zero Fill of truncated US data.
#
#  us_nuszf_rist.ft2      IST NUS Zero Fill of original US data, retaining original time-domain points.
#  us50_nuszf_rist.ft2    IST NUS Zero Fill of truncated US data, retaining original time-domain points.
#
#  us_nuszf_smile.ft2     SMILE NUS Zero Fill of original US data.
#  us50_nuszf_smile.ft2   SMILE NUS Zero Fill of truncated US data.
#
#  nus50_ist.ft2          IST of NUS data without NUS Zero Fill.
#  nus50_nuszf_ist.ft2    IST of NUS data with NUS Zero Fill.
#
#  nus50_rist.ft2         IST of NUS data without NUS Zero Fill, retaining original time-domain points.
#  nus50_nuszf_rist.ft2   IST of NUS data with NUS Zero Fill, retaining original time-domain points.
#
#  nus50_nuszf_smile.ft2  SMILE reconstruction of NUS data, with NUS Zero Fill included by default.
#
# Notes: 
#  1. IST and SMILE are reconstruction methods for NUS data.
#  2. NUS Zero Fill is NUS-style extrapolation of time-domain data; it can be applied to both
#     NUS data and conventional US data.
#  3. NMRPipe's implementation of IST generates a spectrum by default, effectively replacing the
#     original time-domain data. It includes an option to adjust the result by retaining all of
#     the original time-domain points. 
#  4. SMILE always retains the original time-domain points, and always performs NUS Zero Fill.

#
# Processing arguments common to all schemes; Results are scaled to max 100.0 for easier comparison.
# Zero filling is adjusted in each scheme so that all spectra are the same size.

