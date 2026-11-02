# ComfyUI H3 Texture Lab

[English](README.md) · [实验详记](docs/EXPERIMENTS.zh-CN.md)

主要用途是 **Character Swap：一段源视频＋一张人物参考图**。视频提供动作、姿态、场景；图片提供替换人物的身份、面孔和发型。

## 来源

本项目从 Akatz AI / Akatz Labs 的 **Character Swap V1.5 · H3 Relay 2.0** 工作流改造而来。人物替换 LoRA、训练和原始工作流属于原作者成果。

- [原始模型及工作流](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
- [原始 Relay V2 JSON](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/examples/H3-Character-Swap-Relay-V2.json)
- [H3 Relay](https://github.com/akatz-ai/h3-relay)
- [Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)：原生 latent 续接、Refine Blend、Join Match 来自此项目，本项目调用这些节点。

## 现在用哪个

导入 `examples/H3-TextureLab-CharacterSwap-480.json`，需要更大画面再使用 `H3-TextureLab-CharacterSwap-768.json`。这两份都是新建工作流，不覆盖已有工作流。可选自动清理使用新增的 `H3-TextureLab-CharacterSwap-480-CacheCleanup.json` 或 `...768-CacheCleanup.json`，开关默认关闭。安装 H3 Relay、Motion Context 和本节点包；将 `examples/inputs` 的测试图片和视频复制到 ComfyUI/input，或在工作流中选择自己的输入。模型另行下载。

它们保留源视频 SAM 人物遮罩和原音轨，使用原作者 **Character Swap V1.5＋FL2V Turbo 8-step**，强度均为 1，`er_sde / normal`，CFG 1。初始短测 56 帧、24fps，39 帧窗口、22 帧原生 latent 上下文，固定种子 194731。准备视频的时长改为 0 可处理完整输入，迭代次数从帧数自动计算。480/768 是原版像素预算预设，实际尺寸取决于输入。

FFN 分块设为 2048 以适配小显存，fused modulation 关闭。遮罩外保留源 latent；上下文节点保留原噪声遮罩。首帧不强制贴成参考肖像的构图。音轨使用源音频；全局 Join Match 关闭，避免改动受保护的背景。

## SSD 写入与缓存清理

**磁盘窗口缓存以额外磁盘写入换取较低的内存占用。** 每个采样窗口（包括重叠上下文）都会以未压缩的 float32 视频/音频 safetensors 写入 `<输出目录>/H3-TextureLab-Cache/<run-id>/`。manifest 和合成视频也会产生写入，最终保存视频还可能再生成一份输出文件。因此，总写入量会超过最终压缩 MP4 的大小，并随分辨率、时长、重叠量和重复运行次数增加。

如果输出/缓存目录位于 SSD，这些写入会计入 SSD 写入负载并带来磨损。本项目尚未测量长期 SSD 寿命影响。请预留足够空闲空间，并在选择输出磁盘时考虑写入负载；磁盘速度也可能影响缓存和导出耗时。

**默认保留缓存。** 如需自动清理，使用新增的 `H3TextureSaveVideo`（Save Video & Optional Cache Cleanup）节点，连接 Finish Video 的视频输出与同一个 End Loop 的 state。新增示例 `H3-TextureLab-CharacterSwap-480-CacheCleanup.json` 和 `...768-CacheCleanup.json` 已接好；将 **`cleanup_cache`** 开关改成 **true** 才启用，默认 **false**。

该节点先将 MP4 保存到缓存目录之外，逐帧解码检查保存的视频和音频流，核对帧数、画布、帧率及音频流是否保留，成功后才删除本次完整任务中已识别的缓存文件。返回的视频指向最终保存文件，清理后仍能播放。保存/校验失败，或在清理前取消，均保留缓存；发现未知文件、符号链接或 manifest 不匹配时会跳过清理并在 report 中说明。不会清理其他任务目录或模型缓存。

请把它作为本次任务的最终保存节点。如果其他分支、续接、恢复或后续重导出仍需要这些 latent 或合成视频，请保持关闭；本保存节点成功不代表其他无关分支也成功。旧工作流不变，仍可继续使用标准 Save Video。示例保留原生 Start Loop，后续运行会建立新的缓存状态。

也可在确认最终视频已保存到缓存目录之外，且无运行中的任务需要这些文件后手动清理。清理只能释放空间，**不能撤销已经发生的 SSD 写入和磨损**。

## 实验范围

此前的静态、20 秒、40 秒和全身运动样片属于**单图 Ref2V 组件实验**，不能当作视频＋图片人物替换的质量验证。Character Swap 的独立端到端测试另行记录在实验文档中。

磁盘缓存能避免累计保存所有生成帧，也减少历史帧反复 VAE 解码/编码。它不能保证模型本身永不改变肤色、纹理或身份。最终仍按窗口解码，并非一次性把 40 秒画面全部装入显存。

单图实验使用 Ref2V Turbo，Character Swap 使用 FL2V Turbo，两者配方不同。`normal` 与 `simple` 的末尾 sigma 不同，refine 不能混为一谈。音频导出支持源音轨；暂不解码生成的音频 latent。

完整安装、参数和复现命令见 [English README](README.md)，实验过程、成功和失败结果见 [中文实验记录](docs/EXPERIMENTS.zh-CN.md)。代码 GPL-3.0-only；模型及上游代码保留各自许可。发布内容不包含私人原视频、私人工作流、用户拍摄的来源截图或模型权重。
