#!/bin/bash
python inpaint.py "./demo/spiderman.png" "spiderman" "Bat man from DC universe." --output_dir="./spiderman_demo" --verbose
python inpaint.py "./spiderman_demo/spiderman.png" "spiderman" "Iron man from Marvel universe." --output_dir="./spiderman_demo2" --verbose