#!/bin/bash

for core in 16 24 32
do

echo " "
echo "Core: $core"
echo " "

for bit in 32
do

echo " "
echo "BIT: $bit"
echo " "

for skew in 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
echo " "
echo "SKEW: $skew"
echo " "

# No optimizations

echo -n "No OPT"

for k in 1 10 100 1000
do
tmp=0
for i in {1..10}
do
# Run the program and capture its output
echo "./lsb_64 $k $core 2 $bit 0 1 $skew" &>> res_chiplet_skew_out_file.txt
output=$(./lsb_32 $k $core 2 $bit 0 1 $skew 2>&1)
echo "$output" &>> res_chiplet_skew_out_file.txt
echo " " &>> res_chiplet_skew_out_file.txt

# Extract the GB / sec value from the last line
gbps=$(echo "$output" | grep -oP '(?<=\().*?(?= GB / sec)' | tail -n 1)

tmp=$(echo "$tmp + $gbps" | bc)
done

average_gbps=$(echo "scale=2; $tmp / 10" | bc)

echo -n ",$average_gbps"
done

echo ""

echo -n "Chiplet aware"

for k in 1 10 100 1000
do
tmp=0
for i in {1..10}
do
# Run the program and capture its output
echo "./chiplet_lsb_32 $k $core 1 $bit 0 1 $skew" &>> res_chiplet_skew_out_file.txt
output=$(./chiplet_lsb_32 $k $core 1 $bit 0 1 $skew 2>&1)
echo "$output" &>> res_chiplet_skew_out_file.txt
echo " " &>> res_chiplet_skew_out_file.txt

# Extract the GB / sec value from the last line
gbps=$(echo "$output" | grep -oP '(?<=\().*?(?= GB / sec)' | tail -n 1)

tmp=$(echo "$tmp + $gbps" | bc)
done

average_gbps=$(echo "scale=2; $tmp / 10" | bc)

echo -n ",$average_gbps"
done

echo ""

done
done
done






