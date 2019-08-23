(ns clojure-mxnet-autoencoder.model-specs
  (:require [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.ndarray :as ndarray-api]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [clojure-mxnet-autoencoder.viz :as viz]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]))

(def data-dir "data/")
;;; just deal with single numbers here
(def batch-size 1)

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

(defn discriminate [image]
  (-> (m/forward discriminator-model {:data [image]})
      (m/outputs)
      (ffirst)
      (ndarray/argmax-channel)
      (ndarray/->vec)
      (first)
      (int)))

(defn generate [label]
  (-> (m/forward generator-model {:data [(ndarray/array [label] [batch-size])]})
      (m/outputs)
      (ffirst)))


(comment 
  (def my-test-batch (mx-io/next test-data))
  (def my-test-image (first (mx-io/batch-data my-test-batch)))
  (ndarray/shape my-test-image)
  (viz/im-sav {:title "test-discriminator-image" :output-path "results/" :x (ndarray/reshape my-test-image [batch-size 1 28 28])})

  (discriminate my-test-image) ;=> 6
  (def generated-test-image (generate 3))
  (viz/im-sav {:title "generated-image" :output-path "results/" :x (ndarray/reshape generated-test-image [batch-size 1 28 28])})

  (s/def ::mnist-number (s/and int? #(<= 0 % 9)))
  (def x (gen/fmap #(do (generate %))
                   (s/gen ::mnist-number)))
  (def gen-images (->> (gen/sample x)
                       (map ndarray/copy)))
  (doall (map discriminate gen-images))

  (doall (map-indexed (fn [i image]
                        (viz/im-sav {:title (str "result-image-" i)
                                     :output-path "results/"
                                     :x (ndarray/reshape image [batch-size 1 28 28])}))
                      gen-images))
  
  (def y (apply ndarray/stack gen-images))
  (ndarray/shape y)
  (viz/im-sav {:title "generated-image-stack" :output-path "results/" :x (ndarray/reshape y [10 1 28 28])})

  ;;;;;;;;;
  (s/def ::mnist-number (s/and int? #(<= 0 % 9)))

  (gen/sample (s/gen ::mnist-number)) ;=> (0 1 0 3 5 3 7 5 0 1)


  (s/def ::mnist-image
    (s/with-gen
      #(s/valid? ::mnist-number (discriminate %))
      #(gen/fmap (fn [n]
                   (do (ndarray/copy (generate n))))
                 (s/gen ::mnist-number))))

  (def gen-images (gen/sample (s/gen ::mnist-image)))
  (mapv discriminate gen-images) ;=>  [3 3 0 0 1 8 1 9 0 0]
  (viz/im-sav {:title "generated-mnist-images"
               :output-path "results/"
               :x (-> (apply ndarray/stack gen-images)
                      (ndarray/reshape [10 1 28 28]))})


  (s/valid? ::mnist-image my-test-image) ;=> true
  (s/conform ::mnist-image my-test-image)

  ;;;; macro

  (defmacro def-model-spec [spec-key spec discriminate-fn generate-fn]
    `(s/def ~spec-key
       (s/with-gen
         #(s/valid? ~spec (~discriminate-fn %))
         #(gen/fmap (fn [n#]
                      (do (ndarray/copy (~generate-fn n#))))
                    (s/gen ~spec)))))

  (macroexpand-1 `(def-model-spec
                    ::mnist-image2
                    ::mnist-number
                    discriminate
                    generate ))
  (def-model-spec ::mnist-image2 ::mnist-number discriminate generate)

  (s/valid? ::mnist-image2 my-test-image)
  (def gen-images2 (gen/sample (s/gen ::mnist-image2)))
  (viz/im-sav {:title "generated-mnist-images2"
               :output-path "results/"
               :x (-> (apply ndarray/stack gen-images2)
                      (ndarray/reshape [10 1 28 28]))})
  

  ;;;;;;; evens






  )



