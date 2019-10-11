(ns clojure-mxnet-autoencoder.model-specs
  (:require [clojure-mxnet-autoencoder.viz :as viz]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

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

(defmacro def-model-spec [spec-key spec discriminate-fn generate-fn]
    `(s/def ~spec-key
       (s/with-gen
         #(s/valid? ~spec (~discriminate-fn %))
         #(gen/fmap (fn [n#]
                      (do (ndarray/copy (~generate-fn n#))))
                    (s/gen ~spec)))))

(defn test-model-spec [spec test-value]
  (let [gen-images (gen/sample (s/gen spec))]
    (do (viz/im-sav {:title (str "sample-" (name spec))
                     :output-path "results/"
                     :x (-> (apply ndarray/stack gen-images)
                            (ndarray/reshape [(count gen-images) 1 28 28]))}))
   {:spec (name spec)
    :valid? (s/valid? spec test-value)
    :sample-values (mapv discriminate gen-images)}))


(comment 
  (def my-test-batch (mx-io/next test-data))
  (def my-test-image (first (mx-io/batch-data my-test-batch)))
  (ndarray/shape my-test-image)
  (viz/im-sav {:title "test-discriminator-image" :output-path "results/" :x (ndarray/reshape my-test-image [batch-size 1 28 28])})

  (discriminate my-test-image) ;=> 6
  (def generated-test-image (generate 3))
  (viz/im-sav {:title "generated-image" :output-path "results/" :x (ndarray/reshape generated-test-image [batch-size 1 28 28])})

  ;;;;;;;;;
  (s/def ::mnist-number (s/and int? #(<= 0 % 9)))
  (s/valid? ::mnist-number 3) ;=> true
  (s/valid? ::mnist-number 11) ;=> false
  (gen/sample (s/gen ::mnist-number)) ;=> (0 1 0 3 5 3 7 5 0 1)


  (s/def ::mnist-image
    (s/with-gen
      #(s/valid? ::mnist-number (discriminate %))
      #(gen/fmap (fn [n]
                   (do (ndarray/copy (generate n))))
                 (s/gen ::mnist-number))))

  (s/valid? ::mnist-image my-test-image) ;=> true
  (s/conform ::mnist-image my-test-image)


  (test-model-spec ::mnist-image my-test-image)
  ;; {:spec "mnist-image",
  ;;  :valid? true,
  ;;  :sample-values [0 0 0 0 2 7 0 1 0 1]}


  ;;;;;;; evens

  (def-model-spec ::even-mnist-image
    (s/and ::mnist-number even?)
    discriminate
    generate)

  (test-model-spec ::even-mnist-image my-test-image)

  ;; {:spec "even-mnist-image",
  ;;  :valid? true,
  ;;  :sample-values [0 0 2 0 0 0 2 0 0 8]}

   ;;;;;;; odds

  (def-model-spec ::odd-mnist-image
    (s/and ::mnist-number odd?)
    discriminate
    generate)

  (test-model-spec ::odd-mnist-image my-test-image)

  ;; {:spec "odd-mnist-image",
  ;;  :valid? false,
  ;;  :sample-values [1 1 1 3 9 7 3 1 1 3]}


  ;;; odd and over 4

  (def-model-spec ::odd-over-2-mnist-image
    (s/and ::mnist-number odd? #(> % 2))
    discriminate
    generate)

  (test-model-spec ::odd-over-2-mnist-image my-test-image)

  ;; {:spec "odd-over-2-mnist-image",
  ;;  :valid? false,
  ;;  :sample-values [3 3 3 3 3 9 3 3 3 7]}


  )



