(ns clojure-mxnet-autoencoder.discriminator
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
(def
  train-data
  (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                     :label (str data-dir "train-labels-idx1-ubyte")
                     :input-shape [784]
                     :flat true
                     :batch-size batch-size
                     :shuffle true}))

(def
  test-data
  (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                     :label (str data-dir "t10k-labels-idx1-ubyte")
                     :input-shape [784]
                     :batch-size batch-size
                     :flat true
                     :shuffle true}))

(def input (sym/variable "input"))
(def output (sym/variable "input_"))

(defn get-symbol []
  (as-> input data
    ;; encode
    (sym/fully-connected "encode1" {:data data :num-hidden 100})
    (sym/activation "sigmoid1" {:data data :act-type "sigmoid"})

    ;; encode
    (sym/fully-connected "encode2" {:data data :num-hidden 50})
    (sym/activation "sigmoid2" {:data data :act-type "sigmoid"})

    ;;; this last bit changed from autoencoder
    ;;output
    (sym/fully-connected "result" {:data data :num-hidden 10})
    (sym/softmax-output {:data data :label output})))

(def data-desc (first (mx-io/provide-data-desc train-data)))
;;; so we are actually using the label now too
(def label-desc (first (mx-io/provide-label-desc train-data)))

;;; this part needs to change the label shapes to actually be the real labels
(def model (-> (m/module (get-symbol) {:data-names ["input"] :label-names ["input_"]})
               (m/bind {:data-shapes [(assoc data-desc :name "input")]
                        :label-shapes [(assoc label-desc :name "input_")]})
               (m/init-params {:initializer  (initializer/uniform 1)})
               (m/init-optimizer {:optimizer (optimizer/adam {:learning-rage 0.001})})))

;;; we use accuracy instead of mse
(def my-metric (eval-metric/accuracy))

(defn train [num-epochs]
  (doseq [epoch-num (range 0 num-epochs)]
    (println "starting epoch " epoch-num)
    (mx-io/do-batches
     train-data
     (fn [batch]
       ;;; here we make sure to use the label now for forward and update-metric
       (-> model
           (m/forward {:data (mx-io/batch-data batch) :label (mx-io/batch-label batch)})
           (m/update-metric my-metric (mx-io/batch-label batch))
           (m/backward)
           (m/update))))
    (println "result for epoch " epoch-num " is " (eval-metric/get-and-reset my-metric))))

(comment

  (def my-batch (mx-io/next train-data))
  (def images (mx-io/batch-data my-batch))
  (viz/im-sav {:title "originals" :output-path "results/" :x (ndarray/reshape (first images) [100 1 28 28])})


 ;;; before training
  (def my-test-batch (mx-io/next test-data))
  (def test-images (mx-io/batch-data my-test-batch))
  (viz/im-sav {:title "test-images" :output-path "results/" :x (ndarray/reshape (first test-images) [100 1 28 28])})
  (def preds (m/predict-batch model {:data test-images} ))
  (->> preds
       first
       (ndarray/argmax-channel)
       (ndarray/->vec)
       (take 10))
 ;=> (1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

  (train 3)

;; starting epoch  0
;; result for epoch  0  is  [accuracy 0.8178333]
;; starting epoch  1
;; result for epoch  1  is  [accuracy 0.93405]
;; starting epoch  2
;; result for epoch  2  is  [accuracy 0.9528]



 ;;; after training
  (def preds (m/predict-batch model {:data test-images} ))
  (->> preds
       first
       (ndarray/argmax-channel)
       (ndarray/->vec)
       (take 10))

 ;=>  (7.0 5.0 7.0 7.0 3.0 9.0 7.0 4.0 1.0 7.0)


  ;;; save model
  (m/save-checkpoint model {:prefix "model/discriminator" :epoch 2})


  )
