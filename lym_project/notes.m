deep_conv_classification_alt28.py   nucleus-level classification, segmentation center
deep_conv_classification_alt29.py   nucleus-level classification, segmentation center, beta=-inf, 60x60 input
deep_conv_classification_alt30.py   nucleus-level classification, segmentation center, 60x60 input
deep_conv_classification_alt31.py   nucleus-level classification, click center, 60x60 input
deep_conv_classification_alt32.py   deep_conv_classification_alt29.py without Yi's features
deep_conv_classification_alt33.py   deep_conv_classification_alt29.py, click center, without Yi's features
deep_conv_classification_alt34.py   deep_conv_classification_alt32.py with added batch_norm
deep_conv_classification_alt37.py   deep_conv_classification_alt34.py with added batch_norm, normal beta
deep_conv_classification_alt38.py   deep_conv_classification_alt34.py train on all data
deep_conv_classification_alt39.py   deep_conv_classification_alt36.py using deep_conv_ae_spsparse_alt28.py (dense mask)
deep_conv_classification_alt43.py   deep_conv_classification_alt37.py, autoencoder with dense mask
deep_conv_classification_alt44.py   deep_conv_classification_alt37.py, fully supervised

deep_conv_classification_alt35.py   train on nucleus-level data, test on patch-level data

deep_conv_classification_alt27.py   recovered baseline
deep_conv_classification_alt36.py   patch-level better baseline
deep_conv_classification_alt40.py   patch-level better baseline, on 5% of the training set
deep_conv_classification_alt41.py   patch-level better baseline, no cae initialization
deep_conv_classification_alt42.py   patch-level better baseline, no cae initialization, dense mask

deep_conv_classification_alt45.py   retrain using anne's annotations
deep_conv_classification_alt46.py   retrain using fernando's annotations
deep_conv_classification_alt47.py   retrain using john's annotations
deep_conv_classification_alt48.py   retrain using anne's annotations, less epochs
deep_conv_classification_alt49.py   retrain using fernando's annotations, less epochs
deep_conv_classification_alt50.py   retrain using john's annotations, less epochs
deep_conv_classification_alt51.py   fully supervised CNN
deep_conv_classification_alt52.py   retrain using anne's annotations, deterministic=False
deep_conv_classification_alt53.py   retrain using john's annotations, deterministic=False
deep_conv_classification_alt54.py   deep_conv_classification_alt52.py + smaller learning rate

deep_conv_classification_alt60.py   retrain using anne's annotations on BRCA
deep_conv_classification_alt61.py   retrain using anne&john's AGREED annotations on BRCA
deep_conv_classification_alt62.py   retrain using anne&john's AGREED annotations on 10 BRCA, 10 LUAD
deep_conv_classification_alt63.py   retrain using anne&john's AGREED annotations on 10 BRCA, 10 LUAD, constant beta
deep_conv_classification_alt64.py   fully supervised CNN, retrain using anne&john's AGREED annotations on 10 BRCA, 10 LUAD, constant beta

deep_conv_ae_spsparse_alt21.py      auto-encoder baseline
deep_conv_ae_spsparse_alt22.py      no pooling layers
deep_conv_ae_spsparse_alt23.py      no pooling layers, more layers
deep_conv_ae_spsparse_alt24.py      no pooling layers, more layers, dense mask
deep_conv_ae_spsparse_alt25.py      inverse unpooling
deep_conv_ae_spsparse_alt26.py      no pooling layer, fewer filters
deep_conv_ae_spsparse_alt27.py      no pooling layer, fewer filters, dense mask
deep_conv_ae_spsparse_alt28.py      deep_conv_ae_spsparse_alt21.py + dense mask
deep_conv_ae_spsparse_alt29.py      deep_conv_ae_spsparse_alt26.py with encoding vector (not feasible)
deep_conv_ae_spsparse_alt30.py      deep_conv_ae_spsparse_alt21.py with encoding vector
deep_conv_ae_spsparse_alt31.py      deep_conv_ae_spsparse_alt26.py, higher SPR thres, tighter SoftThres
deep_conv_ae_spsparse_alt32.py      deep_conv_ae_spsparse_alt26.py, higher SPR thres, softer SoftThres
deep_conv_ae_spsparse_alt33.py      deep_conv_ae_spsparse_alt26.py, alpha=0.1
deep_conv_ae_spsparse_alt34.py      rebuilt deep_conv_ae_spsparse_alt26.py: less global neurons, output sample sparsity, etc.
deep_conv_ae_spsparse_alt35.py      deep_conv_ae_spsparse_alt34.py, more filters
deep_conv_ae_spsparse_alt36.py      no pooling layers, less dense
deep_conv_ae_spsparse_alt37.py      no pooling layers, less dense, more global encoding neurons
deep_conv_ae_spsparse_alt39.py      more deeper, more thiner, large batch size
deep_conv_ae_spsparse_alt40.py      deeper, thiner
deep_conv_ae_spsparse_alt41.py      deeperer, thinerer

deep_conv_ae_spsparse_alt42.py      cae on 40X v1
deep_conv_ae_spsparse_alt43.py      cae on 40X v2
deep_conv_ae_spsparse_alt44.py      cae on 40X v3
deep_conv_ae_spsparse_alt45.py      cae on 40X v4
deep_conv_ae_spsparse_alt46.py      deep_conv_ae_spsparse_alt21.py + dense
deep_conv_ae_spsparse_alt47.py      deep_conv_ae_spsparse_alt21.py + 16% sparse

