
#!/usr/bin/env bash
#
# runcode.sh for post‐resonance (choose grid, a and M)
#
# Usage: ./runcode.sh -g <grid> -a <a_value> -M <M_value>
#
# Example: ./runcode.sh -g 64 -a 0.5 -M 0.05

set -euo pipefail

# defaults (if any)
KG_BASE="/mnt/vast-standard/home/niki.suckau/u10965/analysis/scalarverse/KG"
SP_BASE="/mnt/vast-standard/home/niki.suckau/u10965/analysis/scalarverse/SP/post_resonance"
TEMPLATE="inputs_template"
TMP_TEMPLATE="inputs_for_preprocess"
FINAL="inputs_final"
EXE="./AxNyx3d.gnu.MPI.ex"   # your SP executable
NP=48                        # number of MPI ranks; adjust as needed

usage() {
  cat <<EOF
Usage: $0 -g <grid> -a <a_value> -M <M_value>

  -g  grid size (32,64,128,256,512)
  -a  value of a (e.g. 0.5)
  -M  value of M (e.g. 0.05)
EOF
  exit 1
}

# parse arguments
while getopts "g:a:M:" opt; do
  case $opt in
    g) grid=$OPTARG ;;
    a) a_val=$OPTARG ;;
    M) M_val=$OPTARG ;;
    *) usage ;;
  esac
done

# require all three
if [[ -z "${grid-}" || -z "${a_val-}" || -z "${M_val-}" ]]; then
  echo "Error: missing required arguments." >&2
  usage
fi

# 1) locate the specified resonance output directory
resdir="${KG_BASE}/Output_${grid}_a${a_val}_M${M_val}"
if [[ ! -d "$resdir" ]]; then
  echo "Error: specified resonance output not found: $resdir" >&2
  exit 1
fi
echo "→ Found resonance dir: $resdir"

# 2) inside it, find the highest-numbered plt directory
pltdir=$(ls -d "${resdir}"/plt* 2>/dev/null \
         | sort -V \
         | tail -n1)
if [[ ! -d "$pltdir" ]]; then
  echo "Error: no plt* directory found in $resdir" >&2
  exit 1
fi
echo "→ Using initial data from: $pltdir"

# 3) build and prepare your SP output dir
OUTDIR="${SP_BASE}/Output_${grid}_a${a_val}_M${M_val}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
echo "→ Writing post-resonance output to: $OUTDIR"

# 4) rewrite KGinitDirName in a temp template
sed -E "s#^KG\.KGinitDirName.*#KG.KGinitDirName = ${pltdir}#" \
    "$TEMPLATE" > "$TMP_TEMPLATE"

# 5) preprocess the template (fills in ${OUTPUT_BASE} elsewhere)
python preprocess_input.py "$TMP_TEMPLATE" "$FINAL" \
       '{"OUTPUT_BASE": "'"$OUTDIR"'"}'

# 6) run the post-resonance simulation
# mpirun -np $NP "$EXE" "$FINAL"
# for single‐CPU debug, uncomment:
mpirun -np 1 "$EXE" "$FINAL"

