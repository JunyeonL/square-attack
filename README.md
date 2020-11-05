# Square Attack: a query-efficient black-box adversarial attack via random search

**Maksym Andriushchenko\*, Francesco Croce\*, Nicolas Flammarion, Matthias Hein**

**EPFL, University of Tübingen**

**Paper:** [https://arxiv.org/abs/1912.00049](https://arxiv.org/abs/1912.00049)

\* denotes equal contribution


## Running the code

Untarget Lp infinite Sqaure attack 동작
 **1. Imagenet validation set 다운로드**  
    - 다운로드 소스 : `https://academictorrents.com/collection/imagenet-2012`  
    - Imagenet 폴더를 만들고 그 안에 압축파일 해제 `tar -xvf ILSVRC2012_img_val.tar`  
    - Imagenet 소스 폴더를 Label별로 정리하기위한 스크립트 실행  
    - `wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash`
    
 **2. data.py 변경하기**
    - data.py 파일안에 IMAGENET_PATH를 다운받은 Imagenet 폴더로 변경
    
 **3. Square Attack 코드는 GPU 기준으로 작성되었기 때문에 CPU만 사용할 경우 코드 수정이 필요함.**
    - attack.py의 `os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` 를 `os.environ["CUDA_VISIBLE_DEVICES"] = '-1'`로 변경  
    - models.py 안의 `DataParallel` 부분 삭제  
    - models.py 안의 `torch.device('cuda')`를 `torch.device('cpu')`로 변경  
   
 
`attack.py` is the main module that implements the Square Attack, see the command line arguments there.
The main functions which implement the attack are `square_attack_linf()` and `square_attack_l2()`.

In order to run the untargeted Linf Square Attack on ImageNet models from the PyTorch repository you need to specify a correct path 
to the validation set (see `IMAGENET_PATH` in `data.py`) and then run:
- ``` python attack.py --attack=square_linf --model=pt_vgg       --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000 ```
- ``` python attack.py --attack=square_linf --model=pt_resnet    --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000 ```
- ``` python attack.py --attack=square_linf --model=pt_inception --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000 ```

Note that eps=12.75 is then divided by 255, so in the end it is equal to 0.05.

For performing targeted attacks, one should use additionally the flag `--targeted`, use a lower `p`, and specify more 
iterations `--n_iter=100000` since it usually takes more iteration to achieve a misclassification to some particular, 
randomly chosen class.

The rest of the models have to downloaded first (see the instructions below), and then can be evaluated in the following way:

Post-averaging models:
- ``` python attack.py --attack=square_linf --model=pt_post_avg_cifar10  --n_ex=1000 --eps=8.0 --p=0.3 --n_iter=20000 ```
- ``` python attack.py --attack=square_linf --model=pt_post_avg_imagenet --n_ex=1000 --eps=8.0 --p=0.3 --n_iter=20000 ```

Clean logit pairing and logit squeezing models:
- ``` python attack.py --attack=square_linf --model=clp_mnist   --n_ex=1000  --eps=0.3   --p=0.3 --n_iter=20000 ```
- ``` python attack.py --attack=square_linf --model=lsq_mnist   --n_ex=1000  --eps=0.3   --p=0.3 --n_iter=20000 ```
- ``` python attack.py --attack=square_linf --model=clp_cifar10 --n_ex=1000  --eps=16.0  --p=0.3 --n_iter=20000 ```
- ``` python attack.py --attack=square_linf --model=lsq_cifar10 --n_ex=1000  --eps=16.0  --p=0.3 --n_iter=20000 ```

Adversarially trained model (with only 1 restart; note that the results in the paper are based on 50 restarts):
- ``` python attack.py --attack=square_linf --model=madry_mnist_robust --n_ex=10000 --eps=0.3 --p=0.8 --n_iter=20000 ```

The L2 Square Attack can be run similarly, but please check the recommended hyperparameters in the paper (Section B of the supplement)
and make sure that you specify the right value `eps` taking into account whether the pixels are in [0, 1] or in [0, 255] 
for a particular dataset dataset and model.
For example, for the standard ImageNet models, the correct L2 eps to specify is 1275 since after division by 255 it will become 5.0.



## Saved statistics
In the folder `metrics`, we provide saved statistics of the attack on 4 models: Inception-v3, ResNet-50, VGG-16-BN.\
Here are simple examples how to load the metrics file.

### Linf attack
To print the statistics from the last iteration:
```
metrics = np.load('metrics/2019-11-10 15:57:14 model=pt_resnet dataset=imagenet n_ex=1000 eps=12.75 p=0.05 n_iter=10000.metrics.npy')
iteration = np.argmax(metrics[:, -1])  # max time is the last available iteration
acc, acc_corr, mean_nq, mean_nq_ae, median_nq, avg_loss, time_total = metrics[iteration]
print('[iter {}] acc={:.2%} acc_corr={:.2%} avg#q={:.2f} avg#q_ae={:.2f} med#q_ae={:.2f} (p={}, n_ex={}, eps={}, {:.2f}min)'.
      format(n_iters+1, acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, p, n_ex, eps, time_total/60))
```

Then one can also create different plots based on the data contained in `metrics`. For example, one can use `1 - acc_corr`
to plot the success rate of the Square Attack at different number of queries.

### L2 attack
In this case we provide the number of queries necessary to achieve misclassification (`n_queries[i] = 0` means that the image `i` was initially misclassified, `n_queries[i] = 10001` indicates that the attack could not find an adversarial example for the image `i`).
To load the metrics and compute the success rate of the Square Attack after `k` queries, you can run:
```
n_queries = np.load('metrics/square_l2_resnet50_queries.npy')['n_queries']
success_rate = float(((n_queries > 0) * (n_queries <= k)).sum()) / (n_queries > 0).sum()
```


## Models
Note that in order to evaluate other models, one has to first download them and move them to the folders specified in 
`model_path_dict` from `models.py`:
- [Clean Logit Pairing on MNIST](https://oc.cs.uni-saarland.de/owncloud/index.php/s/w2yegcfx8mc8kNa)
- [Logit Squeezing on MNIST](https://oc.cs.uni-saarland.de/owncloud/index.php/s/a5ZY72BDCPEtb2S)
- [Clean Logit Pairing on CIFAR-10](https://oc.cs.uni-saarland.de/owncloud/index.php/s/odcd7FgFdbqq6zL)
- [Logit Squeezing on CIFAR-10](https://oc.cs.uni-saarland.de/owncloud/index.php/s/EYnbHDeMbe4mq5M)
- MNIST, Madry adversarial training: run `python madry_mnist/fetch_model.py secret`
- MNIST, TRADES: download the [models](https://drive.google.com/file/d/1scTd9-YO3-5Ul3q5SJuRrTNX__LYLD_M) and see their [repository](https://github.com/yaodongyu/TRADES)
- [Post-averaging defense](https://github.com/YupingLin171/PostAvgDefense/blob/master/trainedModel/resnet110.th): the model can be downloaded directly from the repository

For the first 4 models, one has to additionally update the paths in the `checkpoint` file in the following way: 
```
model_checkpoint_path: "model.ckpt"
all_model_checkpoint_paths: "model.ckpt"
```



## Requirements
- PyTorch 1.0.0
- Tensorflow 1.12.0



## Contact
Do you have a problem or question regarding the code?
Please don't hesitate to open an issue or contact [Maksym Andriushchenko](https://github.com/max-andr) or 
[Francesco Croce](https://github.com/fra31) directly.


## Citation
```
@article{ACFH2020square,
  title={Square Attack: a query-efficient black-box adversarial attack via random search},
  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas and Hein, Matthias},
  conference={ECCV},
  year={2020}
}
```
