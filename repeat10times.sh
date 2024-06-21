#!/bin/bash


echo -n "$1 $2 $3 $4 $5"

tmp=0
for i in {1..10}
do
# Run the program and capture its output
output=$(./$1 $2 $3 $4 $5 $6 $7 $8 2>&1)

# Extract the GB / sec value from the last line
gbps=$(echo "$output" | grep -oP '(?<=\().*?(?= GB / sec)' | tail -n 1)

# Print only the GB / sec value
tmp=$(echo "$tmp + $gbps" | bc)
done

average_gbps=$(echo "scale=2; $tmp / 10" | bc)

echo ", $average_gbps"
