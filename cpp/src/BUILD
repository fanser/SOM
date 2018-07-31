cc_binary(
    name="test",
    srcs=[]
)

cc_library(
    name="utils",
    srcs=["src/utils/*.cc"],
    export_incs=["src/utils"],
    deps=[
      "cpp3rdlib/folly:v2016.08.22.00@//cpp3rdlib/folly:folly",
    ],
    optimize=["O3"],
)

cc_library(
    name="matcher",
    incs=["src/matcher"],
    srcs=["src/matcher/*.cc", "src/matcher/*/*.cc"],
    export_incs=["src/matcher"],
    deps=[
      ":utils",
      "cpp3rdlib/folly:v2016.08.22.00@//cpp3rdlib/folly:folly",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_core",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_imgproc",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_highgui",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_imgcodecs",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_flann",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_features2d",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_calib3d",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_shape",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_xfeatures2d",
      "cpp3rdlib/Eigen@//cpp3rdlib/Eigen:Eigen"
    ],
    optimize=["O3"],
)

thrift_library(
    name = "horus_thrift",
    export_incs=["src/thrift"],
    srcs = [
        "src/thrift/horus.thrift",
    ],
    deps = [
        "cpp3rdlib/thrift@//cpp3rdlib/thrift:thrift",
    ],
)

cc_library(
    name="feature",
    srcs=["src/feature/*/*.cc", "src/feature/*/*/*.cc"],
    export_incs=["src/feature"],
    deps=[
      ":utils",
      ":horus_thrift",
      "cpp3rdlib/folly:v2016.08.22.00@//cpp3rdlib/folly:folly",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_core",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_imgproc",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_highgui",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_imgcodecs",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_video",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_videoio",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_features2d",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_ml",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_objdetect",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_shape",
      "cpp3rdlib/opencv:3.2.0_cpu_nonfree@//cpp3rdlib/opencv:opencv_xfeatures2d",
      "cpp3rdlib/ffmpeg@//cpp3rdlib/ffmpeg:avcodec",
      "cpp3rdlib/Eigen@//cpp3rdlib/Eigen:Eigen",
      "lab/lab_cpp3rdlib@//lab/lab_cpp3rdlib:freeimage",
      "cpputil/consul@//cpputil/consul:consul"
    ],
    optimize=["O3"],
)

cc_plugin(
    name="dolly",
    srcs=["src/dolly/dolly.cc"],
    deps=[
      ":feature",
      ":matcher",
    ],
    optimize=["O3"],
    bundle_path="lib"
)
