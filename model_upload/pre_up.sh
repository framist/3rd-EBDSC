#!/bin/sh

echo "提交准备"

cd ..
# 获取当前日期，格式为 YYYY-MM-DD
current_date=$(date +%Y-%m-%d)
# 忽略文件
zip -r "up_$current_date.zip" model -x \
"model/temp*" \
"model/tmp*" \
"model/__pycache__/*" \
"model/models/__pycache__/*" \
"model/data/*" \
"*.csv" \
"model/.*" &&

echo "完成 -> up_$current_date.zip"