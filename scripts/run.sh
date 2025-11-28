unit=$1
time_range="0 120"

echo "数据检查"
uv run ./DataCheck.py \
    --time_range $time_range \
    -u "$unit" \
    -r \
    -v | tee $unit/check.log

echo "数据标定"
uv run ./Calibration.py \
    --time_range $time_range \
    -u "$unit" \
    -r \
    -v | tee $unit/calibration.log
