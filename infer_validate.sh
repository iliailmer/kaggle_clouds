for i in {1..5}
do
    python3 infer_validate.py -F $i --model smpunet -A resnet50
done