#!/bin/bash 
## Test
#python inpaint.py "./demo/Distracted_Boyfriend.png" "A woman with red shirt" "An Asian, rich, old grandma." --output_dir="./demo_output"

#python inpaint.py "./demo/spiderman.png" "spiderman" "Bat man from DC universe." --output_dir="./spiderman_demo"
#python inpaint.py "./demo/no_face.png" "Black man" "Big fat American woman" --output_dir="./demo_no_face"

## Demo 1: style transfer
# python inpaint.py "./demo/hide_the_pain_harold.png" "The face" "The same face but with sun glasses." --output_dir="./demo_coffee" --verbose
# python inpaint.py "./demo/hide_the_pain_harold.png" "The face" "The face of Xi Jinping, the Chinese president." --output_dir="./demo_coffee" --verbose
# python inpaint.py "./demo/hide_the_pain_harold.png" "The head" "The armor of iron man form Marvel universe" --output_dir="./demo_coffee" --verbose
# python inpaint.py "./demo/hide_the_pain_harold.png" "The head" "The face of Barack Obama, the former U.S. president. His is bold." --output_dir="./demo_coffee" --verbose

# Meme
# python inpaint.py "./demo/president_linkedin.png" "The suit of the woman" "Black leather jacket." --output_dir="./demo_president" --verbose
#python inpaint.py "./demo/walk.png" "The white suit and pants." "Pretty evening dress." --output_dir="./demo_president" --verbose

# Series
# python inpaint.py "./demo/Distracted_Boyfriend.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-2.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-3.png" "The man" "Adolf Hitler with a shocking look." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-4.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-5.png" "The man" "Adolf Hitler during a speech." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-6.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-7.png" "The woman" "Adolf Hitler" --output_dir="./demo_bf" --verbose
# python inpaint.py "./demo/Distracted_Boyfriend-8.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose

# python inpaint.py "./demo/Distracted_Boyfriend-X.png" "The man" "Adolf Hitler." --output_dir="./demo_bf" --verbose


## Demo 2-1: view synthesis + style transfer
#python inpaint.py "./demo/synthesized_oldman.png" "The face" "Thw same face but with sun glasses." --.output_dir="./demo_coffee" --verbose
#python inpaint.py "./demo/synthesized_oldman.png" "The head" "The face of Xi Jinping, the Chinese president." --output_dir="./demo_coffee" --verbose
#python inpaint.py "./demo/synthesized_oldman.png" "The head" "The armor of iron man form Marvel universe" --output_dir="./demo_coffee" --verbose

#python inpaint.py "./demo/yelling_at_cat.png" "The yelling woman" "Man shouting angrily." --output_dir="./demo_yelling_at_cat" --verbose
#python inpaint.py "./demo/winnie_the_pooh.png" "The head of pooh" "Xi Jinping, the Chinese president, sleeping" --output_dir="./demo_pooh" --verbose
#python inpaint.py "./demo/black.png" "The people, no-hand" "John Cena, wrestler, smile, looking at camera, hairless" --output_dir="./demo_black" --verbose

# python inpaint.py "./demo/hide_the_pain_harold.png" "The man" "The iron man form Marvel universe" --output_dir="./demo_coffee" --verbose
#python inpaint.py "./demo/I_want_all.png" "The cheek of old man" "Xi Jinping, the Chinese president" --output_dir="./demo_I_want_all" --verbose

#python inpaint.py "./demo/read.png" "Tom the cat" "Tom the cat smiling, holding a tea cup." --output_dir="./demo_read" --verbose
#python inpaint.py "./demo/look.png" "The newspaper" "nautical map" --output_dir="./demo_look" --verbose
#python inpaint.py "./demo/no_face.png" "The food" "Computer devices, RAMs. and GPUs" --output_dir="./demo_no_face"

python inpaint.py "./demo/Distracted_Boyfriend.png" "The front girl" "Spiderman" --output_dir="./demo_bf" --verbose

# python inpaint.py "./demo/kung_fu.png" "The book" "a pad of paper." --output_dir="./demo_kungfu" --verbose
# python inpaint.py "./demo/kung_fu.png" "The man" "An university old professor with glasses." --output_dir="./demo_kungfu" --verbose
