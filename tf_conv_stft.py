import tensorflow as tf
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import tensorflow.contrib.slim as slim
import librosa
import json


class Conv_STFT:
    def __init__(self, n_fft:int, hop_length:int, pad_end=True, window=None, **kwargs):
        """
        n_fft: Frame length and n_fft is constrainted to be same here.
        hop_length:
        pad_end: Whether to pad the end of signals with zeros when the provided frame length and 
                 step produces a frame that lies partially past its end.
        """

        filters = n_fft//2 + 1
        kernel_size = n_fft
        stride = hop_length

        cos_conv_kernel = np.zeros(shape=[kernel_size, 1, filters], dtype=np.float32)
        sin_conv_kernel = np.zeros(shape=[kernel_size, 1, filters], dtype=np.float32)
        for k in range(filters):
            for n in range(kernel_size):
                cos_conv_kernel[n][0][k] = np.cos(-2.0 * np.pi * k * n / n_fft).astype(np.float32)
                sin_conv_kernel[n][0][k] = np.sin(-2.0 * np.pi * k * n / n_fft).astype(np.float32)  

        # print(np.shape(cos_conv_kernel), np.shape(sin_conv_kernel))

        self.real_conv = tf.keras.layers.Conv1D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=stride,
                                                padding="valid",
                                                use_bias=False,
                                                kernel_initializer=tf.constant_initializer(cos_conv_kernel),
                                                trainable=False,
                                                **kwargs)
        self.imag_conv = tf.keras.layers.Conv1D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=stride,
                                                padding="valid",
                                                use_bias=False,
                                                kernel_initializer=tf.constant_initializer(sin_conv_kernel),
                                                trainable=False,
                                                **kwargs)

        self.pad_end = pad_end
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, x):
        '''
        x: 
            shape: [batch, None]
        return: 
            stft[batch, time, frequency]
        '''
        x = tf.convert_to_tensor(np.asarray(x))
        x_shape = tf.shape(x)
        # x_batch = x_shape[0]
        x_length = x_shape[-1]
        with tf.control_dependencies([tf.debugging.assert_non_negative(x_length - 1)]):
            if self.pad_end:
                n_frame = tf.ceil(tf.maximum(x_length - self.n_fft, 0) / self.hop_length) + 1 # n_frame >= 1
            else:
                n_frame = tf.floor(tf.maximum(x_length - self.n_fft, -1) / self.hop_length) + 1 # n_frame >= 0
        with tf.control_dependencies([tf.debugging.assert_positive(n_frame)]):
            fin_x_length = tf.cast((n_frame - 1) * self.hop_length + self.n_fft, tf.int32)
        if self.pad_end: # x_length <= fin_x_length
            new_x = tf.pad(x, [[0,0], [0, fin_x_length-x_length]])
        else: # x_length > fin_x_length
            new_x = x[:, :fin_x_length]

        x = tf.expand_dims(new_x, -1)

        real = self.real_conv(x)
        imag = self.imag_conv(x)
        return tf.complex(real, imag)

if __name__ == "__main__":
    wav, sr = sf.read("p265_002.wav")


    ## conv stft
    conv_stft = Conv_STFT(256, 128, name='conv_stft')
    stft = conv_stft([wav])

    mag = tf.abs(stft)
    var = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(var, print_info=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mag_np = sess.run(mag)
    print(np.shape(mag_np))
    plt.pcolormesh(np.log(mag_np[0].T+0.01))
    plt.colorbar()
    plt.title("Conv STFT(DFT)")
    plt.show()
    plt.close()

    ## test use librosa
    tmp = librosa.core.stft(wav,
                            n_fft=256,
                            hop_length=128,
                            window=np.ones([256]))
    tmp = np.absolute(tmp)
    plt.pcolormesh(np.log(tmp+0.01))
    plt.colorbar()
    plt.title("librosa STFT(FFT)")
    plt.show()
    plt.close()

    # error spectrum
    err_mag = tmp[:,:-1] - mag_np[0].T
    plt.pcolormesh(err_mag)
    plt.colorbar()
    plt.title("Error spectrum")
    plt.show()
    plt.close()
