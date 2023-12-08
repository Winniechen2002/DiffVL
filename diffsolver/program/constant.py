# Constant weight, We can add small, big, middle into naming of the obj, and connect the weight with the size
# Tell gpt what object we don't need to move, we want them keep there places
# Shape carving need a larger weight

fix_shape_weight = 100.
fix_place_weight = 100.
big_object_weight = 2.
samll_object_weight = 20.
keep_place_object_weight = 40.
emd_weight = 10.
no_break_weight = 0.2
touch_weight = 0.5
away_weight = 0.1
shape_carving_weight = 100.
lift_up_weight = 10.
