(defproject clojure-mxnet-autoencoder "0.1.0-SNAPSHOT"
  :description "Clojure MXNet AutoEncoder"
  :plugins [[lein-cljfmt "0.5.7"]]
  :repositories [["vendredi" {:url "https://repository.hellonico.info/repository/hellonico/"}]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "1.5.0"]
                 [origami "4.0.0-3"]])
