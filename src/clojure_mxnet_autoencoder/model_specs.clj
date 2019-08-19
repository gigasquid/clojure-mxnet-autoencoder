(ns clojure-mxnet-autoencoder.model-specs
  (:require [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [clojure-mxnet-autoencoder.viz :as viz]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [clojure.spec.alpha :as s]))

(def data-dir "data/")
;;; just deal with single numbers here
(def batch-size 100)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "./get_mnist_data.sh"))

(def
  test-data
  (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                     :label (str data-dir "t10k-labels-idx1-ubyte")
                     :input-shape [784]
                     :batch-size batch-size
                     :flat true
                     :shuffle true}))

(def data-desc
  (first
   (mx-io/provide-data-desc test-data)))
(def label-desc
  (first
   (mx-io/provide-label-desc test-data)))


(def discriminator-model
  (-> (m/load-checkpoint {:prefix "model/discriminator" :epoch 2})
      (m/bind {:for-training false
               :data-shapes [(assoc data-desc :name "input")]
               :label-shapes [(assoc label-desc :name "input_")]})
      (m/init-params {:initializer  (initializer/uniform 1)})))

(def generator-model
  (-> (m/load-checkpoint {:prefix "model/generator" :epoch 2})
      (m/bind {:for-training false
               :data-shapes [(assoc label-desc :name "input")]
               :label-shapes [(assoc data-desc :name "input_")]})
      (m/init-params {:initializer  (initializer/uniform 1)})))

(defn discriminate [images]
  (-> (m/forward discriminator-model {:data [image]})
      (m/outputs)
      (ffirst)
      (ndarray/argmax-channel)
      (ndarray/->vec)))

(defn generate [labels]
  (-> (m/forward generator-model {:data [(ndarray/array labels [batch-size])]})
      (m/outputs)
      (ffirst)))


(comment 
  (def my-test-batch (mx-io/next test-data))
  (def my-test-images (first (mx-io/batch-data my-test-batch)))
  (ndarray/shape my-test-images)
  (viz/im-sav {:title "test-discriminator-images" :output-path "results/" :x (ndarray/reshape my-test-images [batch-size 1 28 28])})

  (discriminate my-test-images)
  (def generated-test-images (generate (repeatedly 100 #(rand-int 9))))
  (viz/im-sav {:title "generated-images" :output-path "results/" :x (ndarray/reshape generated-test-images [batch-size 1 28 28])})



  
  
  (discriminate my-test-image) ;=> 6.0


  (s/def ::even-image #(-> (discriminate %)
                           (int)
                           (even?)))

  (s/valid? ::even-image my-test-image) ;=> true

  (s/def ::odd-image #(-> (discriminate %)
                          (int)
                          (odd?)))
  (s/valid? ::odd-image my-test-image) ;=> false


  (def my-test-label (first (mx-io/batch-label my-test-batch)))
  )

   


