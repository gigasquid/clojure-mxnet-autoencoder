(ns clojure-mxnet-autoencoder.auto-encoder
  (:require [clojure-mxnet-autoencoder.viz :as viz]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]))

(def data-dir "data/")
(def batch-size 100)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "./get_mnist_data.sh"))


;;; Load the MNIST datasets
(def train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                   :label (str data-dir "train-labels-idx1-ubyte")
                                   :input-shape [784]
                                   :label-shape [10]
                                   :flat true
                                   :batch-size batch-size
                                   :shuffle true}))

(def test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                  :label (str data-dir "t10k-labels-idx1-ubyte")
                                  :input-shape [784]
                                  :batch-size batch-size
                                  :flat true
                                  :shuffle true}))
(def output (sym/variable "input_"))

(defn get-symbol []
  (as-> (sym/variable "input") data
    ;; encode
    (sym/fully-connected "encode1" {:data data :num-hidden 100})
    (sym/activation "sigmoid1" {:data data :act-type "sigmoid"})

    ;; encode
    (sym/fully-connected "encode2" {:data data :num-hidden 50})
    (sym/activation "sigmoid2" {:data data :act-type "sigmoid"})

    ;; decode
    (sym/fully-connected "decode1" {:data data :num-hidden 50})
    (sym/activation "sigmoid3" {:data data :act-type "sigmoid"})

    ;; decode
    (sym/fully-connected "decode2" {:data data :num-hidden 100})
    (sym/activation "sigmoid4" {:data data :act-type "sigmoid"})

    ;;output
    (sym/fully-connected "result" {:data data :num-hidden 784})
    (sym/activation "sigmoid5" {:data data :act-type "sigmoid"})

    (sym/linear-regression-output {:data data :label output})))

(def data-desc (first (mx-io/provide-data-desc train-data)))

(def model (-> (m/module (get-symbol) {:data-names ["input"] :label-names ["input_"]})
               (m/bind {:data-shapes [(assoc data-desc :name "input")]
                        :label-shapes [(assoc data-desc :name "input_")]})
               (m/init-params {:initializer  (initializer/uniform 1)})
               (m/init-optimizer {:optimizer (optimizer/adam {:learning-rage 0.001})})))

(def my-metric (eval-metric/mse))

(defn train [num-epochs]
  (doseq [epoch-num (range 0 num-epochs)]
    (println "starting epoch " epoch-num)
    (mx-io/do-batches
     train-data
     (fn [batch]
       (-> model
           (m/forward {:data (mx-io/batch-data batch) :label (mx-io/batch-data batch)})
           (m/update-metric my-metric (mx-io/batch-data batch))
           (m/backward)
           (m/update))))
    (println "result for epoch " epoch-num " is " (eval-metric/get-and-reset my-metric))))

(comment

  (mx-io/provide-data train-data)
  (mx-io/provide-label train-data)
  (mx-io/reset train-data)
  (def my-batch (mx-io/next train-data))
  (def images (mx-io/batch-data my-batch))
  (ndarray/shape (ndarray/reshape (first images) [100 1 28 28]))
  (viz/im-sav {:title "originals" :output-path "results/" :x (ndarray/reshape (first images) [100 1 28 28])})


 ;;; before training
 (def my-test-batch (mx-io/next test-data))
 (def test-images (mx-io/batch-data my-test-batch))
 (def preds (m/predict-batch model {:data test-images} ))
 (viz/im-sav {:title "before-training-preds" :output-path "results/" :x (ndarray/reshape (first preds) [100 1 28 28])})

 (train 3)


 ;;; after training
 (def my-test-batch (mx-io/next test-data))
 (def test-images (mx-io/batch-data my-test-batch))
 (def preds (m/predict-batch model {:data test-images} ))
 (viz/im-sav {:title "after-training-preds" :output-path "results/" :x (ndarray/reshape (first preds) [100 1 28 28])})


  )
