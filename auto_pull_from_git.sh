#!/bin/bash
DIR_PaperMod=$1
if [ -z "$1" ]; then
	DIR_PaperMod="/home/ubuntu/git/PaperMod/"
fi
# Set the maximum number of loops
max_loops=4

# Counter to keep track of the loops
counter=0

while [ $counter -lt $max_loops ]; do
    # Your task or command here
    echo "Running your task... Loop: $counter"
    cd $DIR_PaperMod && git pull origin master
    
    # Increment the counter
    ((counter++))

    # Sleep for 30 seconds
    sleep 15
done

