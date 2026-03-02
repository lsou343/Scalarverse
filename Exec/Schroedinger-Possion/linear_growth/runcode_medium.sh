#!/bin/bash

TEMPLATE="inputs_template_medium"
FINAL="inputs_final_medium"
EXE="./AxNyx3d.gnu.MPI.ex"

# NSTEP=$(grep -E "^\s*max_step\s*=" "$TEMPLATE" | sed -E 's/.*=\s*([^# ]+).*/\1/')
#
# if [ -z "$NSTEP" ]; then
#     echo "Error: Could not extract nstep from $TEMPLATE"
#     exit 1
# fi

# FDORPS=$(grep -E "^\s*max_step\s*=" "$TEMPLATE" | sed -E 's/.*=\s*([^# ]+).*/\1/')

OUTDIR="/mnt/vast-standard/home/niki.suckau/u10965/analysis/scalarverse/SP/Linearpertur/PS/medium"
# OUTDIR="Output"
# Delete old output dir
rm -r "$OUTDIR"

# Make sure the output directory exists
mkdir -p "$OUTDIR"

# Preprocess the template
python preprocess_input.py "$TEMPLATE" "$FINAL" '{"OUTPUT_BASE": "'"$OUTDIR"'"}'

# Run the simulation
# mpirun -np 1 "$EXE" "$FINAL"
mpirun -np 64 "$EXE" "$FINAL"

