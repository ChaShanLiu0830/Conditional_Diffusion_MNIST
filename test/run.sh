

# list=("0.50" "1.00" "2.00")
list=("2.00")


# Loop through the list
for item in "guide_${list[@]}"
do
    echo "Current item: $item"
    for i in {1..26}
        do
            echo "Iteration: $i"
            python classifier_exp.py --is_train --is_inference --diffusion_timestamp $i --exp_name $item
        done
done


