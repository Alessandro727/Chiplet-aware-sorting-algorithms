#!/bin/bash

for core in 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
do

echo " "
echo "Core: $core"
echo " "

for bit in 16 32 64
do

echo " "
echo "BIT: $bit"
echo " "

# No optimizations

echo -n "No OPT"

for k in 1 10 100 1000
do
tmp=0
for i in {1..10}
do
# Run the program and capture its output
echo "./lsb_64 $k $core 2 $bit" &>> res_chiplet_out_file2.txt
output=$(./lsb_64 $k $core 2 $bit 2>&1)
echo "$output" &>> res_chiplet_out_file2.txt
echo " " &>> res_chiplet_out_file2.txt

# Extract the GB / sec value from the last line
gbps=$(echo "$output" | grep -oP '(?<=\().*?(?= GB / sec)' | tail -n 1)

tmp=$(echo "$tmp + $gbps" | bc)
done

average_gbps=$(echo "scale=2; $tmp / 10" | bc)

echo -n ",$average_gbps"
done

echo ""

echo -n "Chiplet aware"

t=$(($core-1))

for k in 1 10 100 1000
do 
tmp=0
for i in {1..10}
do
# Run the program and capture its output
echo "./chiplet_lsb_64 $k $core 1 $bit" &>> res_chiplet_out_file2.txt
output=$(./chiplet_lsb_64 $k $core 1 $bit 2>&1)
echo "$output" &>> res_chiplet_out_file2.txt
echo " " &>> res_chiplet_out_file2.txt

# Extract the GB / sec value from the last line
gbps=$(echo "$output" | grep -oP '(?<=\().*?(?= GB / sec)' | tail -n 1)

tmp=$(echo "$tmp + $gbps" | bc)
done

average_gbps=$(echo "scale=2; $tmp / 10" | bc)

echo -n ",$average_gbps"
done

echo ""





