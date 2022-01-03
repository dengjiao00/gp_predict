DROP TABLE IF EXISTS `raw`;

CREATE TABLE `raw` (
  `gp_num` varchar(20)  NOT NULL,
  `result` json DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='股票源数据表';
