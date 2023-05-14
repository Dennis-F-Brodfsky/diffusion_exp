python benchmark.py > /dev/null
for i in {3..14}
do
    pytorch-fid /home/zhulei/diffusion_exp/img/cifar-10 /home/zhulei/diffusion_exp/img/dpm_test_$i --device cuda:3
done
cat logs/task04.out | grep FID > logs/task04.log