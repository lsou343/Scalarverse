#!/bin/bash

TEMPLATE="inputs_template"
FINAL="inputs_final"
EXE="./AxNyx3d.gnu.MPI.ex"

# -------------------------------------------------------------------
# 1) extract & sanitize parameters
# -------------------------------------------------------------------

# only the first entry of amr.n_cell (symmetric box)
grid=$(grep -E "^\s*amr.n_cell" "$TEMPLATE" \
       | sed -E 's/.*=\s*([0-9]+).*/\1/')

# pull raw floats
power_raw=$(grep -E "^\s*KG.power" "$TEMPLATE" \
            | sed -E 's/.*=\s*([^# ]+).*/\1/')
mass_raw=$(grep -E "^\s*KG.MASS" "$TEMPLATE" \
           | sed -E 's/.*=\s*([^# ]+).*/\1/')

# replace dots with 'p' so filenames stay safe
power=${power_raw}  #//./p}

# scale and round mass:
#  - multiply by 5.013256549262001
#  - round to nearest integer
mass_scaled=$(awk "BEGIN { printf \"%.2f\", $mass_raw * 5.013256549262001 }")

# -------------------------------------------------------------------
# 2) build the output directory name
# -------------------------------------------------------------------
# BASE=$(pwd)
BASE="/mnt/vast-standard/home/niki.suckau/u10965/analysis/scalarverse/KG"
BASE=""
# OUTDIR="${BASE}/Output_${grid}_a${power}_M${mass_scaled}"
# OUTDIR="${BASE}/Output_${grid}_a${power}_M${mass_scaled}_Test"
OUTDIR="Output"

# -------------------------------------------------------------------
# 3) clean up, make, preprocess, and run
# -------------------------------------------------------------------
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

python preprocess_input.py "$TEMPLATE" "$FINAL" \
       '{"OUTPUT_BASE": "'"$OUTDIR"'"}'

# mpirun -np 1 "$EXE" "$FINAL"
# mpirun -np 6 "$EXE" "$FINAL"
mpirun -np 12 "$EXE" "$FINAL"
# mpirun -np 128 "$EXE" "$FINAL"

