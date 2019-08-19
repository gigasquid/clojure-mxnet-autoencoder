(ns clojure-mxnet-autoencoder.generator
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
(defonce
  train-data
  (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                     :label (str data-dir "train-labels-idx1-ubyte")
                     :input-shape [784]
                     :flat true
                     :batch-size batch-size
                     :shuffle true}))

(defonce
  test-data (mx-io/mnist-iter
             {:image (str data-dir "t10k-images-idx3-ubyte")
              :label (str data-dir "t10k-labels-idx1-ubyte")
              :input-shape [784]
              :batch-size batch-size
              :flat true
              :shuffle true}))

(def input (sym/variable "input"))
(def output (sym/variable "input_"))

;;; need to add onehot encoding to use the label as input

(defn get-symbol []
  (as-> input data
    (sym/one-hot "onehot" {:indices data :depth 10})
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

(def data-desc
  (first
   (mx-io/provide-data-desc train-data)))
(def label-desc
  (first
   (mx-io/provide-label-desc train-data)))

(def
  model
  ;;; change data shapes to label shapes
  (-> (m/module (get-symbol) {:data-names ["input"] :label-names ["input_"]})
      (m/bind {:data-shapes [(assoc label-desc :name "input")]
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
       ;;; change input to be the label
       (-> model
           (m/forward {:data (mx-io/batch-label batch) :label (mx-io/batch-data batch)})
           (m/update-metric my-metric (mx-io/batch-data batch))
           (m/backward)
           (m/update))))
    (println "result for epoch " epoch-num " is " (eval-metric/get-and-reset my-metric))))

(comment

  (def my-batch (mx-io/next train-data))
  (def images (mx-io/batch-data my-batch))
  (viz/im-sav
   {:title "originals"
    :output-path "results/"
    :x (ndarray/reshape (first images) [100 1 28 28])})


 ;;; before training
  (def my-test-batch (mx-io/next test-data))
  ;;; change to input labels
  (def test-labels (mx-io/batch-label my-test-batch))
  (def preds (m/predict-batch model {:data test-labels} ))
  (viz/im-sav {:title "before-training-preds" :output-path "results/" :x (ndarray/reshape (first preds) [100 1 28 28])})

  (train 3)


 ;;; after training
  (def my-test-batch (mx-io/next test-data))
  (def test-labels (mx-io/batch-label my-test-batch))
  (def preds (m/predict-batch model {:data test-labels} ))
  (viz/im-sav {:title "after-training-preds" :output-path "results/" :x (ndarray/reshape (first preds) [100 1 28 28])})
  (->> test-labels first ndarray/->vec (take 10))
                                        ;=> (7.0 5.0 7.0 7.0 3.0 9.0 7.0 4.0 7.0 7.0)


  ;;; save model
  (m/save-checkpoint model {:prefix "model/generator" :epoch 2})


  )


