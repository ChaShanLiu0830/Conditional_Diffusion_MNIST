

list=("1.00" "2.00")
# list=("2.00")


# Loop through the list
for item in "${list[@]}"
do
    echo "Current item: guide_${item}"
    for i in {1..22}
        do
            echo "Iteration: $i"
            python classifier_exp.py --is_train --is_inference --diffusion_timestamp $i --exp_name "guide_${item}"
        done
done


