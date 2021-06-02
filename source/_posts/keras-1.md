---
categories: Machine Learning
title: Keras 学习笔记(2)
date: 2021-03-13 15:50:03
tags: [Keras, Tensorflow, Python, Machine Learning]
---

1. [RNN (LSTM, GRU) 模型](https://keras.io/zh/layers/recurrent/)
- `lstm1, lstm_h, lstm_c = LSTM(hideen_size, return_sequences=True, return_state=True)(input)`
返回lstm的每层隐状态`lstm1`,最后输出`lstm_h`,最后的单元状态`lstm_c`。

2. [Bidirenctial layer](https://keras.io/api/layers/recurrent_layers/bidirectional/)
- `lstm_out = Bidirectional(LSTM(10, return_sequences=True)(input))`
- 也可以分开写
```
forward_layer = LSTM(10, return_sequences=True)(input)
backward_layer = LSTM(10, activation='relu', return_sequences=True,
                       go_backwards=True)(input)
```
- [使用`return_state`](https://stackoverflow.com/questions/47923370/keras-bidirectional-lstm-seq2seq)
```
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = Bidirectional(LSTM(latent_dim, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
```

3. [mask 使用方法](https://www.sohu.com/a/407812302_500659)
- Masking layer:
```
mask = keras.layers.Masking(mask_value= 0, input_shape=(time_step,feature_size))(input)
lstm_output = keras.layers.LSTM(hidden_size, return_sequences= True)(mask)
```
- Embedding layer:
```
embed = keras.layers.Embedding(vocab_size, embedding_size, mask_zero= True)(input)
lstm_output = keras.layers.LSTM(hidden_size, return_sequences= True)(emded)
```

4. [optimizer](https://keras.io/zh/optimizers/)
- `tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)`
	- `decay`: 学习率衰减
- Learning rate
	- tf 2.0 `tf.keras.optimizers.schedules.LearningRateSchedule`
```
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(200)
custom_adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                 epsilon=1e-9)
```
	- keras: `keras.callbacks.LearningRateScheduler(schedule)`
```
import keras.backend as K
from keras.callbacks import LearningRateScheduler
 
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
 
reduce_lr = LearningRateScheduler(scheduler)
model.fit(train_x, train_y, batch_size=32, epochs=300, callbacks=[reduce_lr])
```
- [Reduce LR On Plateau](https://blog.csdn.net/zzc15806/article/details/79711114)
`keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)`
```

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
model.fit(train_x, train_y, batch_size=32, epochs=300, validation_split=0.1, callbacks=[reduce_lr])
```