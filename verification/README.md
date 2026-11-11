独立验收说明

冻结提交：https://github.com/9873199/pyjwt-jwks-generations/commit/2989f48fe07acf97c54c2098d20e8ade9001cc44
实际解释器：3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]

从产物快照取出相同版本，用environment-reviewed.json记录的Python与依赖执行两个verification目录中的真实命令。回归针对整个tests；独立验收key_rotation_check.py使用本地生成RSA密钥、RS256签名、进程内transport与可控时钟/同步屏障。无需外部JWKS服务或账号。录屏调用同一冻结库，验证控制台单独提供。最早的根目录回归命令路径有误，不能算作有效成绩。

每组目录内的 *-command.json 保留真实参数、工作目录、PYTHONPATH、退出码和原始日志位置；test-results.json 与 JUnit XML 保留完整成绩。跨机器运行时把这些绝对路径映射到同一冻结 checkout 和自己的缓存目录，所有 --basetemp、XML 和构建输出均放在缓存里。依赖版本见 environment-reviewed.json。
这些材料在模型单轮结束后生成，与原轨内测试分开，不代表项目方质检通过。原轨和视频只放各自的飞书附件。


冻结产物提交：https://github.com/9873199/pyjwt-jwks-generations/commit/2989f48fe07acf97c54c2098d20e8ade9001cc44
这些独立验收文件在模型首轮结束后生成，不是模型原轨中的测试。
