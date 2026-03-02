#!/bin/bash

# JUP="/mnt/vast-standard/home/niki.suckau/u10965/analysis/scalarverse_gaussian/Output"
JUP="Output"

# Remove specific files from the output directory
rm -rf "${JUP}"/plt* \
       "${JUP}"/chk* \
       "${JUP}"/runlog \
       "${JUP}"/rholog \
       "${JUP}"/grdlog \
       mem_info.log \
       Backtrace.* \
       Output/outfile-* \
       Output/err*

# rm -rf Output/plt* Output/chk* Output/runlog Output/grdlog mem_info.log Backtrace.* Output/outfile-*
