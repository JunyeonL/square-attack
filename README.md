# Square Attack: a query-efficient black-box adversarial attack via random search

**Maksym Andriushchenko\*, Francesco Croce\*, Nicolas Flammarion, Matthias Hein**

**EPFL, University of Tübingen**

**Paper:** [https://arxiv.org/abs/1912.00049](https://arxiv.org/abs/1912.00049)

\* denotes equal contribution


**Original source codes and descriptions: https://github.com/max-andr/square-attack

## Running the code

## Untarget Lp infinite Sqaure attack 실행하기

 **1. Imagenet validation set 다운로드**  
>    - 다운로드 소스 : `https://academictorrents.com/collection/imagenet-2012`  
>    - Imagenet 폴더를 만들고 그 안에 압축파일 해제 `tar -xvf ILSVRC2012_img_val.tar`  
>    - Imagenet 소스 폴더를 Label별로 정리하기위한 스크립트 실행  
>    - `wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash`
    
 **2. data.py 변경하기**  
>    - data.py 파일안에 IMAGENET_PATH를 다운받은 Imagenet 폴더로 변경
    
 **3. Square Attack 코드는 GPU 기준으로 작성되었기 때문에 CPU만 사용할 경우 코드 수정이 필요함.**  
>    - attack.py의 `os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` 를 `os.environ["CUDA_VISIBLE_DEVICES"] = '-1'`로 변경  
>    - models.py 안의 `DataParallel` 부분 삭제  
>    - models.py 안의 `torch.device('cuda')`를 `torch.device('cpu')`로 변경  

 **4. untarget 공격 실행.**  
>    - `python attack.py --attack=square_linf --model=pt_vgg --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000`  
>    - `python attack.py --attack=square_linf --model=pt_resnet --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000`  
>    - `python attack.py --attack=square_linf --model=pt_inception --n_ex=1000  --eps=12.75 --p=0.05 --n_iter=10000`  
 
 **5. target 공격을 실행하려면 --targeted 옵션으로 지정**  

