import tensorflow as tf
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import tensorflow.contrib.slim as slim
import librosa
import json


class Conv_STFT:
    def __init__(self, n_fft:int, hop_length:int, pad_end=True, window='hann', **kwargs):
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
        if window is not None:
            window_arr = librosa.filters.get_window(window, n_fft)
            cos_conv_kernel *= np.reshape(window_arr, [n_fft, 1, 1])
            sin_conv_kernel *= np.reshape(window_arr, [n_fft, 1, 1])


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
        if type(x) is not tf.Tensor:
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
        return tf.cast(tf.complex(real, imag), tf.complex64)

class Conv_iSTFT:
    def __init__(self, n_fft:int, hop_length:int, forword_window='hann', **kwargs):
        """
        n_fft: Frame length and n_fft is constrainted to be same here.
        hop_length:
        """

        filters = n_fft//2 + 1
        kernel_size = n_fft

        cos_conv_kernel = np.zeros(shape=[filters, kernel_size], dtype=np.float32)
        sin_conv_kernel = np.zeros(shape=[filters, kernel_size], dtype=np.float32)
        for k in range(filters):
            for n in range(kernel_size):
                cos_conv_kernel[k][n] = np.cos(2.0 * np.pi * k * n / n_fft).astype(np.float32)
                sin_conv_kernel[k][n] = np.sin(2.0 * np.pi * k * n / n_fft).astype(np.float32)

        window_arr = self.inverse_stft_window_fn(hop_length, forward_window=forword_window)(n_fft)
        cos_conv_kernel *= np.reshape(window_arr, [1, n_fft])
        sin_conv_kernel *= np.reshape(window_arr, [1, n_fft])

        # print(np.shape(cos_conv_kernel), np.shape(sin_conv_kernel))

        self.istft_kernel_real = tf.convert_to_tensor(cos_conv_kernel)
        self.istft_kernel_imag = tf.convert_to_tensor(sin_conv_kernel)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def inverse_stft_window_fn(self,
                               frame_step: int,
                               forward_window='hann'):
        """Generates a window function that can be used in `inverse_stft`.
        Constructs a window that is equal to the forward window with a further
        pointwise amplitude correction.  `inverse_stft_window_fn` is equivalent to
        `forward_window_fn` in the case where it would produce an exact inverse.
        See examples in `inverse_stft` documentation for usage.
        Args:
            frame_step: An integer scalar. The number of samples to step.
            forward_window_fn: window_fn used in the forward transform, `stft`.
            name: An optional name for the operation.
        Returns:
            A callable that takes a window length and a `dtype` keyword argument and
            returns a `[window_length]` `Tensor` of samples in the provided datatype.
            The returned window is suitable for reconstructing original waveform in
            inverse_stft.
        """

        def inverse_stft_window_fn_inner(frame_length):
            """Computes a window that can be used in `inverse_stft`.
            Args:
            frame_length: An integer scalar `Tensor`. The window length in samples.
            dtype: Data type of waveform passed to `stft`.
            Returns:
            A window suitable for reconstructing original waveform in `inverse_stft`.
            Raises:
            ValueError: If `frame_length` is not scalar, `forward_window_fn` is not a
            callable that takes a window length and a `dtype` keyword argument and
            returns a `[window_length]` `Tensor` of samples in the provided datatype
            `frame_step` is not scalar, or `frame_step` is not scalar.
            """

            # Use equation 7 from Griffin + Lim.
            forward_window_arr = librosa.filters.get_window(forward_window, frame_length)
            denom = np.square(forward_window_arr)
            overlaps = -(-frame_length // frame_step)  # Ceiling division.
            denom = np.pad(denom, [(0, overlaps * frame_step - frame_length)], mode="constant")
            denom = np.reshape(denom, [overlaps, frame_step])
            denom = np.sum(denom, 0, keepdims=True)
            denom = np.tile(denom, [overlaps, 1])
            denom = np.reshape(denom, [overlaps * frame_step])

            return forward_window_arr / denom[:frame_length]
        return inverse_stft_window_fn_inner

    def __call__(self, x):
        '''
        x:
            shape: [batch, time, k], complex
        return:
            singal[batch, None]
        '''
        if type(x) is not tf.Tensor:
            x = tf.convert_to_tensor(np.asarray(x))
        x = tf.expand_dims(x, axis=-2) # [bathc, time, 1, k]

        filters = self.n_fft//2 + 1
        kernel_size = self.n_fft
        real_kernel = tf.reshape(self.istft_kernel_real, [1, 1, filters, kernel_size]) # [1,1,k,n]
        imag_kernel = tf.reshape(self.istft_kernel_imag, [1, 1, filters, kernel_size])

        x_real = tf.real(x)
        x_imag = tf.imag(x)

        s_batch = tf.matmul(x_real, real_kernel) - tf.matmul(x_imag, imag_kernel) # [batch, time, 1, n]
        s_batch = tf.squeeze(s_batch, axis=[-2]) # [batch, time, n]
        s_batch /= filters
        s = tf.signal.overlap_and_add(s_batch, self.hop_length)

        return s

if __name__ == "__main__":
    wav, sr = sf.read("test1.wav")
    wav = wav.astype(np.float32)


    ## conv stft
    
    conv_stft = Conv_STFT(256, 128, window='hann', name='conv_stft')
    stft = conv_stft([wav]) # [batch, time, f]

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

    # istft = Conv_iSTFT(256, 128, forword_window='hann')
    # re_wav_conv = istft(stft)
    # re_wav_conv_np = sess.run(re_wav_conv)
    # print("re_wav_conv", np.shape(re_wav_conv_np))
    # sf.write('re_wav_conv.wav', re_wav_conv_np[0], sr)
    

    '''
    istft = tf.signal.inverse_stft(stft, 256, 128, window_fn=tf.signal.inverse_stft_window_fn(128))
    re_wav_tfistft = tf.signal.overlap_and_add(istft, 128)
    re_wav_tfistft_np = sess.run(re_wav_tfistft)
    print("rela_wav_tfistft shape:", np.shape(re_wav_tfistft_np))
    sf.write("rela_wav_tfistft.wav", re_wav_tfistft_np, sr)

    stft_np = sess.run(stft)[0]
    rela_wav_conv = librosa.core.istft(stft_np.T, hop_length=128, window=np.ones([256]))
    print("rela_wav_conv shape:", np.shape(rela_wav_conv))
    sf.write("rela_wav_conv.wav", rela_wav_conv, sr)
    '''

    # sess = tf.Session()
    stft_tf = tf.signal.stft(wav, 256, 128)
    stft_tf_abs = tf.abs(stft_tf)
    re_wav_tf = tf.signal.inverse_stft(stft, 256, 128, window_fn=tf.signal.inverse_stft_window_fn(128))
    stft_tf_abs_np, re_wav_tf_np = sess.run([stft_tf_abs, re_wav_tf])
    print(np.shape(re_wav_tf_np))
    sf.write('re_wav_coneDFT_tfIFFT.wav', re_wav_tf_np[0], sr)
    # print(np.shape(mag_np))
    # plt.pcolormesh(np.abs(mag_np[0].T[:,:-1]-stft_tf_abs_np.T))
    # plt.colorbar()
    # plt.title("error tf-conv")
    # plt.show()
    # plt.close()
    # plt.plot(np.abs(re_wav_conv_np[0][:40576]-re_wav_tf_np))
    # plt.title("error tf-conv WAV")
    # plt.show()
    # plt.close()
    
    
    '''
    ## test use librosa
    stft_la = librosa.core.stft(wav,
                                n_fft=256,
                                hop_length=128,
                                window='hann')
    tmp = np.absolute(stft_la)
    plt.pcolormesh(np.log(tmp+0.01))
    plt.colorbar()
    plt.title("librosa STFT(FFT)")
    plt.show()
    plt.close()
    re_wav_la = librosa.core.istft(stft_la, hop_length=128, window='hann')
    print("rewav_la shape:", np.shape(re_wav_la))
    sf.write("re_wav_la.wav", re_wav_la, sr)

    # error spectrum
    err_mag = tmp[:,:-1] - mag_np[0].T
    plt.pcolormesh(err_mag)
    plt.colorbar()
    plt.title("Error spectrum")
    plt.show()
    plt.close()
    '''
