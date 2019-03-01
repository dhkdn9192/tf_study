
## Estimators란
----
텐서플로우 프로그래밍을 단순화시키기 위한 high-level API. 다음 기능들을 수행한다:

- training
- evaluation
- prediction

pre-made된 Estimators를 쓸 수도 있고, 직접 Estimators를 만들어 쓸 수도 있다. 그러나 어느쪽이든 공통적으로 <b>tf.estimator.Estimator</b> class를 기반으로 한다 

<p>
    
* 참조1 : https://www.tensorflow.org/guide/estimators
* 참조2 : https://www.tensorflow.org/tutorials/estimators/linear


<p>

## Estimators의 장점
----

* 로컬호스트 또는 분산된 멀티서버 환경에서 모델을 수정하지 않고도 Estimator 기반의 모델을 실행시킬 수 있다 (CPU, GPU, TPU 모두)
* 모델 개발자들이 쉽게 모델을 공유하며 만들 수 있다
* 최신 모델을 high-level의 직관적인 코드로 개발할 수 있다. 즉, low-level 텐서플로우 API보다 더 쉽게 모델을 만들 수 있다.
* tf.keras.layers에 내장되어 있어 커스터마이징이 쉽다
* 자신만의 그래프를 build할 수 있다
* Estimator는 안전하게 분산 반복 학습을 할 수 있으며 다음 요소들을 제어한다:

    - build the graph
    - initialize variables
    - load data
    - handle exceptions
    - create checkpoint files and recover from failures
    - save summaries for TensorBoard


* Estimator로 어플리케이션을 만들 땐 반드시 데이터 입력 파이프라인을 모델로부터 분리해야 한다. 그렇게 해야 다른 데이터셋에 대해서도 쉽게 모델을 실행할 수 있기 때문이다


<p>

## Pre-made Estimators란
----
* pre-made Estimator는 tf.Graph와 tf.Session 객체를 생성 및 관리해준다. (즉 직접 그래프나 세션을 만드느라 고생할 필요가 없다) 
* pre-made Estimator는 서로 다른 모델에 대해서도 최소한의 코드만 수정하면 실행할 수 있도록 해준다. 
    * 예시) tf.estimator.DNNClassifier는 dense, feed-forward 신경망에 기반한 classification 모델을 학습시켜주는 클래스이다.


* pre-made Estimators의 장점
    * 그래프의 서로 다른 부분들 실행할 때 어느 부분을 실행할 지 결정하는데에 도움을 준다
    * 단일 서버 또는 클러스터 모두에서 사용할 수 있다
    * Best practices for event (summary) writing and universally useful summaries.


* pre-made Estimator를 사용하지 않는다면 앞서 말한 기능들을 직접 구현해야 한다


<p>

## Pre-made Estimators 구조
---
#### 1. Write one or more dataset importing functions:
    
* 예시로, 학습셋을 import하는 function과 테스트셋을 import하는 function을 만들 수 있다. 데이터셋을 임포트하는 함수는 반드시 다음의 두 객체를 반환해야 한다
        
    * 키가 feature names이고 값이 (feature data를 포함하는) Tensors인 <b>딕셔너리</b>
    * 하나 이상의 레이블을 포함하는 <b>Tensors</b>





```python
# input_fn style 1
def input_fn(dataset):
    # manipulate dataset, extracting the feature dict and the label
    # (생략)
    return feature_dict, label
```


```python
# input_fn style 2
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"f1": some_numpy_array},      # Input features
      y = some_numpy_array,          # true labels
      batch_size=some_value,
      num_epochs=None,                             # Supply unlimited epochs of data
      shuffle=True)
```

#### 2. Define the feature columns:

* 각각의 tf.feature_column들은 feature의 이름, 타입, 입력 전처리 과정들을 구분한다. 
    * 예시로, 아래의 코드는 integer 또는 float 값을 갖는 3개의 feature column들을 생성한다. 처음 2개 feature column들은 feature들의 이름과 타입을 구분한다. 세 번째 feature column은 raw 데이터의 스케일을 지정하기 위한 람다식을 지정한다


```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                                                    normalizer_fn=lambda x: x - global_education_mean)
```

#### 3. Instantiate the relevant pre-made Estimator:
* For example, here's a sample instantiation of a pre-made Estimator named LinearClassifier:


```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education])
```

#### 4. Call a training, evaluation, or inference method:

* For example, all Estimators provide a train method, which trains a model.


```python
# `input_fn` is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

<p>

## Tutorial
----

tf.estimator에 내장된 Estimator를 사용해보자

* 참조 : https://jhui.github.io/2017/03/14/TensorFlow-Estimator/


```python
import tensorflow as tf
import numpy as np

print(tf.__version__)
```

    /Users/dhkdn9192/venv/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


    1.12.0


#### Define the feature columns

In our example, we define a single feature with name f1


```python
x_feature = tf.feature_column.numeric_column('f1')

# We can use more than one feature. We can even pre normalize the feature with a lambda function:
# n_room = tf.feature_column.numeric_column('n_rooms')
# sqfeet = tf.feature_column.numeric_column('square_feet', normalizer_fn='lambda a: a - global_size')
```

#### input_fn

To import data to the Estimator later, we prepare an input_fn for training, testing and prediction respectively. In each input_fn, we provide all input features and values in x and the true labels in y.


```python
# Training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"f1": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),          # true labels
      batch_size=2,
      num_epochs=None,                             # Supply unlimited epochs of data
      shuffle=True)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"f1": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

# Prediction
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"f1": np.array([8., 9.])},
      num_epochs=1,
      shuffle=False)

```

input_fn in general returns a tuple <b>feature_dict</b> and <b>label</b>. feature_dict is a dict containing the feature names and the feature data, and label contains the true values for all the samples.


```python
print(train_input_fn())
```

    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/python/estimator/inputs/queues/feeding_queue_runner.py:62: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/python/estimator/inputs/queues/feeding_functions.py:500: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    ({'f1': <tf.Tensor 'random_shuffle_queue_DequeueMany:1' shape=(2,) dtype=float64>}, <tf.Tensor 'random_shuffle_queue_DequeueMany:2' shape=(2,) dtype=float64>)



```python
print(predict_input_fn())
```

    {'f1': <tf.Tensor 'fifo_queue_DequeueUpTo:1' shape=(?,) dtype=float64>}


#### Use a pre-built estimator

* TensorFlow comes with many built-in estimator:

    * DNNClassifier
    * DNNLinearCombinedClassifier
    * DNNLinearCombinedRegressor
    * DNNRegressor
    * LinearClassifier
    * LinearRegressor


* To demonstrate the idea, we use the LinearRegressor to model:

    * y=Wx+b


```python
regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': './output', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x11d67bcc0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}


#### Training, validation and testing

Then we run the training, validation and testing with the corresponding input_fn.


```python
# train
regressor.train(input_fn=train_input_fn, steps=2500)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./output/model.ckpt-5000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py:804: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    INFO:tensorflow:Saving checkpoints for 5000 into ./output/model.ckpt.
    INFO:tensorflow:loss = 2.7722535e-10, step = 5001
    INFO:tensorflow:global_step/sec: 999.451
    INFO:tensorflow:loss = 4.5474735e-11, step = 5101 (0.101 sec)
    INFO:tensorflow:global_step/sec: 1363.61
    INFO:tensorflow:loss = 4.0472514e-11, step = 5201 (0.074 sec)
    INFO:tensorflow:global_step/sec: 1248.99
    INFO:tensorflow:loss = 6.895107e-11, step = 5301 (0.080 sec)
    INFO:tensorflow:global_step/sec: 1316.31
    INFO:tensorflow:loss = 4.240519e-11, step = 5401 (0.075 sec)
    INFO:tensorflow:global_step/sec: 1256.49
    INFO:tensorflow:loss = 4.5474735e-11, step = 5501 (0.080 sec)
    INFO:tensorflow:global_step/sec: 1288.77
    INFO:tensorflow:loss = 3.890932e-11, step = 5601 (0.078 sec)
    INFO:tensorflow:global_step/sec: 1348.06
    INFO:tensorflow:loss = 3.1960212e-11, step = 5701 (0.074 sec)
    INFO:tensorflow:global_step/sec: 1409.28
    INFO:tensorflow:loss = 1.546141e-11, step = 5801 (0.071 sec)
    INFO:tensorflow:global_step/sec: 1456.26
    INFO:tensorflow:loss = 4.092726e-12, step = 5901 (0.068 sec)
    INFO:tensorflow:global_step/sec: 1465.96
    INFO:tensorflow:loss = 1.4836132e-11, step = 6001 (0.069 sec)
    INFO:tensorflow:global_step/sec: 1484.83
    INFO:tensorflow:loss = 2.1501023e-11, step = 6101 (0.067 sec)
    INFO:tensorflow:global_step/sec: 1479.51
    INFO:tensorflow:loss = 4.5474735e-13, step = 6201 (0.069 sec)
    INFO:tensorflow:global_step/sec: 1298.88
    INFO:tensorflow:loss = 1.5702994e-11, step = 6301 (0.077 sec)
    INFO:tensorflow:global_step/sec: 1362.44
    INFO:tensorflow:loss = 1.2050805e-11, step = 6401 (0.073 sec)
    INFO:tensorflow:global_step/sec: 1189.97
    INFO:tensorflow:loss = 1.2562396e-11, step = 6501 (0.084 sec)
    INFO:tensorflow:global_step/sec: 1405.94
    INFO:tensorflow:loss = 1.2050805e-11, step = 6601 (0.071 sec)
    INFO:tensorflow:global_step/sec: 1431
    INFO:tensorflow:loss = 1.2050805e-11, step = 6701 (0.070 sec)
    INFO:tensorflow:global_step/sec: 1326.03
    INFO:tensorflow:loss = 1.1269208e-11, step = 6801 (0.076 sec)
    INFO:tensorflow:global_step/sec: 1358.38
    INFO:tensorflow:loss = 1.546141e-11, step = 6901 (0.074 sec)
    INFO:tensorflow:global_step/sec: 1356.04
    INFO:tensorflow:loss = 1.2562396e-11, step = 7001 (0.074 sec)
    INFO:tensorflow:global_step/sec: 1429.88
    INFO:tensorflow:loss = 1.2050805e-11, step = 7101 (0.070 sec)
    INFO:tensorflow:global_step/sec: 1451.63
    INFO:tensorflow:loss = 1.1652901e-11, step = 7201 (0.069 sec)
    INFO:tensorflow:global_step/sec: 1402.19
    INFO:tensorflow:loss = 1.2050805e-11, step = 7301 (0.072 sec)
    INFO:tensorflow:global_step/sec: 1291.06
    INFO:tensorflow:loss = 1.2050805e-11, step = 7401 (0.077 sec)
    INFO:tensorflow:Saving checkpoints for 7500 into ./output/model.ckpt.
    INFO:tensorflow:Loss for final step: 1.2050805e-11.





    <tensorflow.python.estimator.canned.linear.LinearRegressor at 0x11d65beb8>




```python
# evaluate(test)
average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
print(f'>> average_loss : {average_loss}')
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2019-02-21-05:23:46
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./output/model.ckpt-7500
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2019-02-21-05:23:46
    INFO:tensorflow:Saving dict for global step 7500: average_loss = 6.699944e-11, global_step = 7500, label/mean = 11.5, loss = 2.0099833e-10, prediction/mean = 11.499992
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 7500: ./output/model.ckpt-7500
    >> average_loss : 6.699944071764108e-11



```python
# predict
predictions = list(regressor.predict(input_fn=predict_input_fn))
print(f'>> predictions : {predictions}')
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./output/model.ckpt-7500
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    >> predictions : [{'predictions': array([15.499988], dtype=float32)}, {'predictions': array([17.499985], dtype=float32)}]


<p>

## Creating Estimators from Keras models
---
이미 만들어진 Keras 모델을 Estimators로 변환할 수 있다. <b>tf.keras.estimator.model_to_estimator</b>를 이용한다

* 참조 : http://marubon-ds.blogspot.com/2018/01/how-to-convert-keras-model-to.html



```python
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)
```

    1.12.0


#### Load mnist data


```python
# load mnist data for training
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```

    WARNING:tensorflow:From <ipython-input-11-9ce68095cbf9>:2: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-images-idx3-ubyte.gz
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting ./mnist/data/t10k-images-idx3-ubyte.gz
    Extracting ./mnist/data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /Users/dhkdn9192/venv/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.



```python
mnist.train.num_examples
```




    55000




```python
train_data = mnist.train.images[:300]
train_labels = mnist.train.labels[:300]

print(f'train_data.shape : {train_data.shape}')
print(f'train_labels.shape : {train_labels.shape}')
```

    train_data.shape : (300, 784)
    train_labels.shape : (300, 10)


#### Build Keras model


```python
# set model
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, input_dim=8))
model.add(Activation('softmax'))
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                650       
    _________________________________________________________________
    activation (Activation)      (None, 10)                0         
    =================================================================
    Total params: 109,386
    Trainable params: 109,386
    Non-trainable params: 0
    _________________________________________________________________


#### Convert Keras model to Estimator


```python
estimator_model = keras.estimator.model_to_estimator(keras_model=model)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: /var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k
    INFO:tensorflow:Using the Keras model provided.
    INFO:tensorflow:Using config: {'_model_dir': '/var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x11f1b9b38>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
estimator_model
```




    <tensorflow.python.estimator.estimator.Estimator at 0x12b1d2390>



#### Define input_fn


```python
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={model.input_names[0]: train_data.astype(np.float32)},
    y=train_labels.astype(np.float32),
    num_epochs=None,
    shuffle=True)
```

#### Train Estimator


```python
estimator_model.train(input_fn=train_input_fn, steps=50)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Warm-starting with WarmStartSettings: WarmStartSettings(ckpt_to_initialize_from='/var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k/keras/keras_model.ckpt', vars_to_warm_start='.*', var_name_to_vocab_info={}, var_name_to_prev_var_name={})
    INFO:tensorflow:Warm-starting from: ('/var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k/keras/keras_model.ckpt',)
    INFO:tensorflow:Warm-starting variable: dense/kernel; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: dense/bias; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: dense_1/kernel; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: dense_1/bias; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: dense_2/kernel; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: dense_2/bias; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: SGD/iterations; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: SGD/lr; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: SGD/momentum; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: SGD/decay; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable_1; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable_2; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable_3; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable_4; prev_var_name: Unchanged
    INFO:tensorflow:Warm-starting variable: training/SGD/Variable_5; prev_var_name: Unchanged
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into /var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k/model.ckpt.
    INFO:tensorflow:loss = 2.336968, step = 1
    INFO:tensorflow:Saving checkpoints for 50 into /var/folders/vn/ywztlbsn63jfty01rdy5gssr0000gn/T/tmp8i08za4k/model.ckpt.
    INFO:tensorflow:Loss for final step: 1.6971831.





    <tensorflow.python.estimator.estimator.Estimator at 0x12b1d2390>



#### keras.application의 기본 모델에 대한 converting 예시


```python
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
keras_inception_v3
```


```python
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
keras_inception_v3
```


```python
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)
est_inception_v3
```


```python
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={keras_inception_v3.input_names[0]: train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)

```


```python
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```
