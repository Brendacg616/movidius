struct ActionDetectorConfig{
  std::string loc_blob_name{"mbox_loc1/out/conv/flat"};
  /** @brief Name of output blob with detection confidence info */
  std::string detection_conf_blob_name{"mbox_main_conf/out/conv/flat/softmax/flat"};
  /** @brief Prefix of name of output blob with action confidence info */
  std::string action_conf_blob_name_prefix{"out/anchor"};
  /** @brief Name of output blob with priorbox info */
  std::string priorbox_blob_name{"mbox/priorbox"};
  /** @brief Scale paramter for Soft-NMS algorithm */
  float nms_sigma = 0.6f;
  /** @brief Threshold for detected objects */
  float detection_confidence_threshold = 0.4f;
  /** @brief Threshold for recognized actions */
  float action_confidence_threshold = 0.75f;
  /** @brief Scale of action logits */
  float action_scale = 3.0;
  /** @brief Default action class label */
  int default_action_id = 0;
  /** @brief Number of top-score bboxes in output */
  int keep_top_k = 200;
  /** @brief Number of SSD anchors */
  int num_anchors = 4;
  /** @brief Number of actions to detect */
  int num_action_classes = 3;
}
