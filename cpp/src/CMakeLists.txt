#list(APPEND CPP_SRC ../utils/feature_extract.cpp test_feat_extractor.cpp)
SET(DOC_SRC)                     
list(APPEND DOC_SRC ../utils/feature_extract.cpp build_feature_doc.cpp)
SET(SAVER_SRC)                     
list(APPEND SAVER_SRC ../utils/feature_extract.cpp save_feature.cpp)

#add_executable(test_feat_extractor ${CPP_SRC})
#target_link_libraries(test_feat_extractor caffe boost_system opencv_imgproc opencv_highgui opencv_core)

add_executable(doc_builder ${DOC_SRC})
target_link_libraries(doc_builder caffe boost_system opencv_imgproc opencv_highgui opencv_core)

add_executable(feat_saver ${SAVER_SRC})
target_link_libraries(feat_saver caffe boost_system opencv_imgproc opencv_highgui opencv_core)
