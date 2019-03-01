# GAN (Generative Adversarial Network)
----
* 참조1 : https://www.youtube.com/watch?v=odpjk7_tGY0
* 참조2 : https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network


## Branches of ML
----
* Supervised Learning
    * Discriminative Model : 입력 데이터를 각 클래스로 분류하도록 학습하는 모델
* Unsupervised Learning
    * Generative Model : 학습 데이터의 분포를 학습하는 모델
    
    
## Probability Distribution
----
* 확률분포 (probability distribution) :
    * 확률변수가 특정한 값을 가질 확률을 나타내는 함수. 예시로, 주사위를 던졌을 때 나오는 눈에 대한 확률변수가 있을 때, 그 변수의 확률분포는 이산균등분포가 된다. (by 위키백과)
        * 이산 확률 분포
        * 연속 확률 분포


* 생성모델 (generative model)의 목표 :
    * $p_{data}(x)$에 근접하는 $p_{model}(x)$를 찾는 것
        * $p_{data}(x)$ : 실제 이미지의 확률분포
        * $p_{model}(x)$ : 모델이 생성해낸 이미지들의 확률분포
        ![image.png](attachment:image.png)
        
        

## Intuition of GAN
----

* Discriminator
    * real images와 fake images를 구분(binary classification)한다
    * real images 입력시 출력이 1이 되도록 학습
    * fake images 입력시 출력이 0이 되도록 학습
    
    
* Generator
    * 랜덤한 코드(latent code)를 입력받아 Discriminator를 속일 수 있는 fake images를 생성한다
    

