load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//third_party/py/t5/google:build_defs.bzl", "pytype_tf1and2_strict_test")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(
    [
        "LICENSE",
        "trainer.py",
        "trainer_test.py",
        "feature_converters.py",
        "feature_converters_test.py",
        "feature_converters_utils.py",
        "layers.py",
        "layers_test.py",
        "models.py",
        "models_test.py",
        "network.py",
        "network_test.py",
    ],
)

pytype_strict_library(
    name = "feature_converters_utils",
    srcs = [
        "feature_converters_utils.py",
    ],
    deps = [
        "//third_party/py/seqio",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "feature_converters",
    srcs = [
        "feature_converters.py",
    ],
    deps = [
        ":feature_converters_utils",
        "//third_party/py/gin",
        "//third_party/py/seqio",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/t5x:decoding",
        "//third_party/py/t5x:losses",
        "//third_party/py/t5x:models",
        "//third_party/py/t5x:metrics",
        "//third_party/py/t5x:optimizers",
        "//third_party/py/clu:metrics",
        "//third_party/py/flax",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/seqio",
        "//third_party/py/tensorflow",
        "//third_party/py/typing_extensions",
        "//third_party/sentencepiece/src/python:sentencepiece_trainer",
        "//third_party/py/gin",
        ":feature_converters",
    ] + select({
        "//tools/cc_target_os:gce": [],
        "//conditions:default": [
            "//third_party/py/clu:profiling",
        ],
    }),
)

pytype_strict_library(
    name = "trainer",
    srcs = [
        "trainer.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":models",
        "//third_party/py/t5x:metrics",
        "//third_party/py/t5x:models",
        "//third_party/py/t5x:partitioning",
        "//third_party/py/t5x:train_state",
        "//third_party/py/t5x:trainer",
        "//third_party/py/t5x:utils",
        "//third_party/py/absl/logging",
        "//third_party/py/cached_property",
        "//third_party/py/clu:asynclib",
        "//third_party/py/clu:metrics",
        "//third_party/py/clu:values",
        "//third_party/py/clu/data",
        "//third_party/py/clu/metric_writers",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/typing_extensions",
        "//third_party/pybind11_abseil:status",
        "//third_party/sentencepiece/src/python:sentencepiece_trainer",
        ":feature_converters",
    ] + select({
        "//tools/cc_target_os:gce": [],
        "//conditions:default": [
            "//third_party/py/clu:profiling",
        ],
    }),
)

pytype_tf1and2_strict_test(
    name = "feature_converters_test",
    deps = [
        ":feature_converters",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/seqio",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "models_test",
    timeout = "long",
    srcs = ["models_test.py"],
    python_version = "PY3",
    deps = [
        ":models",
        ":network",
        ":trainer",
        "//learning/brain/research/jax:tpu_support",  # buildcleaner:keep
        "//testing/pybase:parameterized",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/seqio",
        "//third_party/py/t5/data:tasks",
        "//third_party/py/t5x:adafactor",
        "//third_party/py/t5x:partitioning",
        "//third_party/py/t5x:test_utils",
        "//third_party/py/t5x:trainer",
        "//third_party/py/t5x:utils",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    python_version = "PY3",
    deps = [
        ":models",
        ":trainer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/chex",
        "//third_party/py/clu:metrics",
        "//third_party/py/clu:values",
        "//third_party/py/clu/metric_writers",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/t5x:metrics",
        "//third_party/py/t5x:optimizers",
        "//third_party/py/t5x:partitioning",
        "//third_party/py/t5x:test_utils",
        "//third_party/py/t5x:train_state",
        "//third_party/py/t5x:trainer",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "network",
    srcs = ["network.py"],
    deps = [
        ":layers",
        "//third_party/py/flax",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "network_test",
    srcs = ["network_test.py"],
    env = {"FLAX_LAZY_RNG": "no"},
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "optonly",
        "requires-dragonfish",
        "requires-net:external",
    ],
    deps = [
        ":network",
        "//learning/brain/research/jax:tpu_support",  # buildcleaner:keep
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/seqio",
        "//third_party/py/t5x:adafactor",
        "//third_party/py/t5x:models",
        "//third_party/py/t5x:test_utils",
    ],
)

py_strict_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    env = {"FLAX_LAZY_RNG": "no"},
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)
