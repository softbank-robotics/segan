## main.pyを動かすとエラー

```
Traceback (most recent call last):
  File "main.py", line 122, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/home/tadashi/.local/lib/python2.7/site-packages/absl/app.py", line 299, in run
    _run_main(main, args)
  File "/home/tadashi/.local/lib/python2.7/site-packages/absl/app.py", line 250, in _run_main
    sys.exit(main(argv))
  File "main.py", line 89, in main
    se_model = SEGAN(sess, FLAGS, udevices)
  File "/home/tadashi/work/segan/model.py", line 118, in __init__
    self.build_model(args)
  File "/home/tadashi/work/segan/model.py", line 134, in build_model
    self.build_model_single_gpu(idx)
  File "/home/tadashi/work/segan/model.py", line 154, in build_model_single_gpu
    self.preemph)
  File "/home/tadashi/work/segan/data_loader.py", line 39, in read_and_decode
    wave = tf.cast(pre_emph(wave, preemph), tf.float32)
  File "/home/tadashi/work/segan/data_loader.py", line 10, in pre_emph
    concat = tf.concat(0, [x0, diff])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/util/dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/ops/array_ops.py", line 1418, in concat
    dtype=dtypes.int32).get_shape().assert_has_rank(0)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 1184, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 1242, in convert_to_tensor_v2
    as_ref=False)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 1297, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/ops/array_ops.py", line 1267, in _autopacking_conversion_function
    return _autopacking_helper(v, dtype, name or "packed")
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/ops/array_ops.py", line 1203, in _autopacking_helper
    return gen_array_ops.pack(elems_as_tensors, name=scope)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/ops/gen_array_ops.py", line 6303, in pack
    "Pack", values=values, axis=axis, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 1770, in __init__
    control_input_ops)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/ops.py", line 1610, in _create_c_op
    raise ValueError(str(e))
ValueError: Dimension 0 in both shapes must be equal, but are 1 and 16383. Shapes are [1] and [16383].
	From merging shape 0 with other shapes. for 'device_0/concat/concat_dim' (op: 'Pack') with input shapes: [1], [16383].
```

### 入力ファイルを指定しなくても、下記エラーがでるので、入力ファイルが
読み込めてない予感

  File "/home/tadashi/work/segan/data_loader.py", line 10, in pre_emph
    concat = tf.concat(0, [x0, diff])

    のx がwavファイルなので、読み込めてないのであたり。

data_load.py L24

def read_and_decode(filename_queue, canvas_size, preemph=0.):
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
            })

            wav_rawが読み取れていない？
            filename_queue が問題。

model.py L148   
    def build_model_single_gpu(self, gpu_idx):
        filename_queue = tf.train.string_input_producer([self.e2e_dataset])
        self.get_wav, self.get_noisy = read_and_decode(filename_queue,
                                        self.canvas_size,
                                        self.preemph)

### 下記データ・セットが読み取れていない。

model.py L103
self.e2e_dataset = args.e2e_dataset

main.py L53
flags.DEFINE_string("e2e_dataset", "data/segan.tfrecords", "TFRecords"
                                                          " (Def: data/"
                                                          "segan.tfrecords.")

        　これが読み取れていない。

* segan.tfrecords が問題。


### test_wavをいれるかいれないかで、trainするか、推論するかの区別

main.pu L95

        if FLAGS.test_wav is None:
            se_model.train(FLAGS, udevices)
        else:
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')

### READMEより

Data
The speech enhancement dataset used in this work (Valentini et al. 2016) can be found in Edinburgh DataShare. However, the following script downloads and prepares the data for TensorFlow format:

```
./prepare_data.sh
```

Or alternatively download the dataset, convert the wav files to 16kHz sampling and set the noisy and clean training files paths in the config file e2e_maker.cfg in cfg/. Then run the script:

```
python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg
```


#### とりあえず pre-emph を０にすると、問題が起きないので


# 次の問題発生

Traceback (most recent call last):
  File "main.py", line 122, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/home/tadashi/.local/lib/python2.7/site-packages/absl/app.py", line 299, in run
    _run_main(main, args)
  File "/home/tadashi/.local/lib/python2.7/site-packages/absl/app.py", line 250, in _run_main
    sys.exit(main(argv))
  File "main.py", line 89, in main
    se_model = SEGAN(sess, FLAGS, udevices)
  File "/home/tadashi/work/segan/model.py", line 118, in __init__
    self.build_model(args)
  File "/home/tadashi/work/segan/model.py", line 134, in build_model
    self.build_model_single_gpu(idx)
  File "/home/tadashi/work/segan/model.py", line 191, in build_model_single_gpu
    do_prelu=do_prelu)
  File "/home/tadashi/work/segan/generator.py", line 191, in __call__
    h_i = tf.concat(2, [z, h_i])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/util/dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/ops/array_ops.py", line 1418, in concat
    dtype=dtypes.int32).get_shape().assert_has_rank(0)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow_core/python/framework/tensor_shape.py", line 995, in assert_has_rank
    raise ValueError("Shape %s must have rank %d" % (self, rank))
ValueError: Shape (2, 100, 8, 1024) must have rank 0

## CUDA及び、tensorflowのversionを落とすことで解決

同時にCUDAを８にする。
tensorflowを0.12にした。
これで解決

tensorflowのversion確認
python -c 'import tensorflow as tf; print(tf.__version__)'   # for Python 2
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3

CUDAダウンロード
https://developer.nvidia.com/cuda-toolkit

CUDAの削除

$ sudo apt purge cuda*
$ sudo apt purge nvidia-cuda-*
$ sudo apt purge libcuda*

or 

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-8.0/bin


## そうするとnvidia-smiではなく、nvcc -cでCUDAが確認できるようになった。

