#!/bin/bash

# Check if the directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

# Get the directory path from the first argument
directory="$1"

# Check if the given path is a directory
if [ ! -d "$directory" ]; then
  echo "Error: $directory is not a valid directory."
  exit 1
fi

# Iterate over files in reverse order
for file in $(ls -1 "$directory" | sort -r); do
  file_path="$directory/$file"
  if [ -f "$file_path" ]; then
    # Replace 'your_command' with the actual command you want to execute
    python tools/train.py --cfg "$file_path"
  fi
done

