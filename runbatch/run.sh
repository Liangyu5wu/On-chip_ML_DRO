#!/bin/bash

for i in {1..500}
do
    echo "Starting experiment $i..."

    python runa.py

    echo "Experiment $i completed."
done

echo "All experiments finished!"
