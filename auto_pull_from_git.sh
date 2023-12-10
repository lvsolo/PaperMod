#!/bin/bash

# Set the maximum number of loops
max_loops=4

# Counter to keep track of the loops
counter=0

while [ $counter -lt $max_loops ]; do
    # Your task or command here
    echo "Running your task... Loop: $counter"
    cd /root/hugo/hugo_themes/themes/PaperMod && git pull origin master
    
    # Increment the counter
    ((counter++))

    # Sleep for 30 seconds
    sleep 15
done

