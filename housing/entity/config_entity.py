from collections import namedtuple

# in this folder jitni bhi chije required hai components ko usko define kis gya hai

DataIngestionconfig= namedtuple("DataIngestionconfig",["dataset_download_url","tgz_download_dir","raw_data_dir","ingested_train_dir","ingested_test_dir"])

DataValidationconfig=namedtuple("DataValidationconfig",["schema_file_path","report_file_path","report_page_file_path"])

DataTransformationconfig=namedtuple("DataTransformationconfig",["add_bedroom_per_room","tranformed_train_dir","transformed_test_dir","preprocessed_object_file_path"])

ModelTrainingconfig=namedtuple("ModelTrainingconfig",["trained_model_file_path","base_accuracy","model_config_file_path"])

ModelEvaluationConfig=namedtuple("ModelEvaluationConfig",["model_evaluation_file_path","time_stamp"])

ModelPusherConfig=namedtuple("ModelPusherConfig",["export_dir_path"])

TrainingPipelineconfig=namedtuple("TrainingPipelineconfig",["artifact_dir"])
